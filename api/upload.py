"""Replenishment calculator: Excel parsing + order quantity logic.

This file is the single source of truth for the calculations. It is deployed as a
Vercel Python function (class `handler`) and is also imported by the local Flask
app (app.py) so both environments run identical logic.

Supported input layouts
-----------------------
1. Legacy export: `Material`, `Vta 01`..`Vta 12` (Vta 01 = most recent full month),
   `Fe. Comp`, `Cant.`, `PVP`, `Vta Prom Mensual`, `Ventas UN`, `Cod. Proveedor Actual`.
2. New "reorder" export: `N° Material`, `Mes12`..`Mes1` (Mes1 = most recent full
   month), `Actual` (current partial month), `Fecha Ult. Compra`,
   `Fecha Ult. Entrega`, `Ult. Cantidad`, `PV`, `Vta Prom Mes`, `Ventas UN Total`,
   `Cod. Proveedor`, `No. Art. Proveedor`, `ABC`, `Grupo`, `Cod. Barra`, `Tecla`.

Both are normalized to the legacy column names internally; extra columns from the
new layout are kept under stable names (see COLUMN_ALIASES).
"""
import datetime as dt
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd

MONTH_COLS = [f"Vta {str(i).zfill(2)}" for i in range(1, 13)]
CURRENT_MONTH_COL = "Vta Actual"

# new header -> canonical header. Month columns are handled separately.
COLUMN_ALIASES = {
    "N° Material": "Material",
    "Nº Material": "Material",
    "No. Material": "Material",
    "Cod. Proveedor": "Cod. Proveedor Actual",
    "Fecha Ult. Compra": "Fe. Comp",
    "Ult. Cantidad": "Cant.",
    "PV": "PVP",
    "Vta Prom Mes": "Vta Prom Mensual",
    "Ventas UN Total": "Ventas UN",
    "ABC": "ABC Proveedor",
    "Actual": CURRENT_MONTH_COL,
}

MATERIAL_HEADERS = {"Material"} | {k for k, v in COLUMN_ALIASES.items() if v == "Material"}

NUMERIC_COLS = [
    "Cant.", "Stock CEDI", "K001 / Q001", "Stock Tiendas", "Stock Total",
    "FOB", "Costo", "PVP", "Vta Prom Mensual", "Ventas UN", CURRENT_MONTH_COL,
] + MONTH_COLS

TEXT_COLS = ["Material", "Marca", "Cod. Proveedor Actual", "Descripcion",
             "No. Art. Proveedor", "ABC Proveedor", "Grupo", "Cod. Barra", "Tecla"]

SERVICE_LEVEL_Z = {
    85: 1.04, 90: 1.28, 92: 1.41, 95: 1.65, 97: 1.88, 98: 2.05, 99: 2.33,
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Velocity tiers (share of total forecast units, cumulative, fastest first)
FAST_SHARE = 0.50
MEDIUM_SHARE = 0.85
SLOW_ABS_RATE = 1.0  # below this many units/month an item is always "Lento"


# ─────────────────────────── multipart parsing ───────────────────────────

def parse_multipart(content_type, body):
    """Parse multipart/form-data without deprecated cgi module."""
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):]
            break
    if not boundary:
        raise ValueError("No boundary found in Content-Type")

    delimiter = b"--" + boundary.encode("utf-8")
    parts = body.split(delimiter)

    fields = {}
    file_data = None
    file_name = None

    for part in parts:
        if part in (b"", b"--\r\n", b"--"):
            continue
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue

        header_section = part[:header_end].decode("utf-8", errors="replace")
        body_section = part[header_end + 4:]
        if body_section.endswith(b"\r\n"):
            body_section = body_section[:-2]

        name = None
        filename = None
        for line in header_section.split("\r\n"):
            if "Content-Disposition" in line:
                for param in line.split(";"):
                    param = param.strip()
                    if param.startswith("name="):
                        name = param.split("=", 1)[1].strip('"')
                    elif param.startswith("filename="):
                        filename = param.split("=", 1)[1].strip('"')

        if name and filename:
            file_data = body_section
            file_name = filename
        elif name:
            fields[name] = body_section.decode("utf-8", errors="replace")

    return fields, file_data, file_name


# ─────────────────────────── excel parsing ───────────────────────────

def _clean_text(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def normalize_columns(headers):
    """Map raw header names to canonical names. Returns list of new names."""
    out = []
    for h in headers:
        h = str(h).strip()
        if h in COLUMN_ALIASES:
            out.append(COLUMN_ALIASES[h])
            continue
        # Mes1..Mes12 (new layout) -> Vta 01..Vta 12 (Mes1 = most recent)
        hl = h.lower().replace(" ", "")
        if hl.startswith("mes") and hl[3:].isdigit():
            n = int(hl[3:])
            if 1 <= n <= 12:
                out.append(f"Vta {str(n).zfill(2)}")
                continue
        out.append(h)
    return out


def parse_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0, header=None)
    except Exception as e:
        raise ValueError(f"Error al leer el archivo Excel: {str(e)}")

    if len(df) == 0:
        raise ValueError("El archivo Excel está vacío")

    header_row = None
    for i in range(min(10, len(df))):
        row_vals = {str(v).strip() for v in df.iloc[i].tolist()}
        if row_vals & MATERIAL_HEADERS:
            header_row = i
            break
    if header_row is None:
        raise ValueError(
            "No se encontró la fila de encabezado con la columna 'Material' o 'N° Material'. "
            "Asegúrese de que el archivo tenga esa columna en las primeras 10 filas."
        )

    headers = normalize_columns(df.iloc[header_row].tolist())
    data = df.iloc[header_row + 1:].copy()
    data.columns = headers
    # drop duplicate-named columns (keep first) to avoid pandas ambiguity
    data = data.loc[:, ~pd.Index(headers).duplicated()]

    data["Material"] = data["Material"].map(_clean_text)
    data = data[data["Material"] != ""]
    if len(data) == 0:
        raise ValueError("No se encontraron datos después de la fila de encabezado")

    for col in NUMERIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        elif col in MONTH_COLS or col == CURRENT_MONTH_COL:
            data[col] = 0.0
    for col in TEXT_COLS:
        if col in data.columns:
            data[col] = data[col].map(_clean_text)
        else:
            data[col] = ""
    return data.reset_index(drop=True)


# ─────────────────────────── classification ───────────────────────────

def classify_demand(monthly_sales):
    """Syntetos-Boylan demand classification by ADI and CV² of non-zero demands."""
    non_zero = [s for s in monthly_sales if s > 0]
    if len(non_zero) == 0:
        return "Sin Demanda", 0, 0
    adi = len(monthly_sales) / len(non_zero)
    if len(non_zero) >= 2:
        mean_nz = np.mean(non_zero)
        std_nz = np.std(non_zero, ddof=1)
        cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0
    else:
        cv2 = 0
    if adi < 1.32 and cv2 < 0.49:
        label = "Suave"
    elif adi < 1.32:
        label = "Errática"
    elif cv2 < 0.49:
        label = "Intermitente"
    else:
        label = "Irregular"
    return label, round(adi, 2), round(cv2, 2)


def croston_forecast(monthly_sales, alpha=0.15):
    """Croston with SBA bias correction. Input is oldest→newest agnostic but we feed
    most-recent-first arrays reversed so smoothing ends on the latest month."""
    series = list(reversed(monthly_sales))  # oldest first
    non_zero = [s for s in series if s > 0]
    if len(non_zero) == 0:
        return 0.0
    if len(non_zero) == 1:
        return non_zero[0] / len(series)
    z_hat = None
    p_hat = 1.0
    q = 0
    for val in series:
        if val > 0:
            if z_hat is None:
                z_hat = val
            else:
                q += 1
                z_hat = alpha * val + (1 - alpha) * z_hat
                p_hat = alpha * q + (1 - alpha) * p_hat
            q = 0
        else:
            q += 1
    if p_hat <= 0:
        return 0.0
    return max(0.0, (z_hat / p_hat) * (1 - alpha / 2))


def moving_average(recent_sales):
    """Average monthly demand over the window (index 0 = most recent).

    Months before the item's first sale in the window are ignored (new products),
    months with zero sales after that count as zero demand. This avoids inflating
    the forecast of slow movers, which the old "sum / months-with-sales" did.
    """
    if not recent_sales:
        return 0.0
    last_sale_idx = None
    for i in range(len(recent_sales) - 1, -1, -1):
        if recent_sales[i] > 0:
            last_sale_idx = i
            break
    if last_sale_idx is None:
        return 0.0
    window = recent_sales[:last_sale_idx + 1]
    return float(sum(window)) / len(window)


def classify_velocity(rates):
    """Tier items by forecast units/month: Rápido / Medio / Lento / '-' (no demand).

    Rápido = fastest items that together make up FAST_SHARE of forecast units.
    Medio  = next items up to MEDIUM_SHARE.
    Lento  = the tail, plus anything below SLOW_ABS_RATE units/month.
    """
    n = len(rates)
    tiers = ["-"] * n
    total = float(sum(r for r in rates if r > 0))
    if total <= 0:
        return tiers
    order = sorted(range(n), key=lambda i: rates[i], reverse=True)
    cum = 0.0
    for i in order:
        r = rates[i]
        if r <= 0:
            continue
        share_before = cum / total  # standard Pareto rule: classify by share reached before this item
        cum += r
        if r < SLOW_ABS_RATE:
            tiers[i] = "Lento"
        elif share_before < FAST_SHARE:
            tiers[i] = "Rápido"
        elif share_before < MEDIUM_SHARE:
            tiers[i] = "Medio"
        else:
            tiers[i] = "Lento"
    return tiers


def calculate_abc(fob_values, unit_sales):
    values = [float(f) * float(u) for f, u in zip(fob_values, unit_sales)]
    total_value = sum(values) if sum(values) > 0 else 1
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    abc = ["C"] * len(values)
    cumulative = 0
    for idx, val in indexed:
        cumulative += val
        pct = cumulative / total_value
        if pct <= 0.80:
            abc[idx] = "A"
        elif pct <= 0.95:
            abc[idx] = "B"
    return abc


def calculate_xyz(items_sales):
    xyz = []
    for sales in items_sales:
        non_zero = [s for s in sales if s > 0]
        if len(non_zero) < 2:
            xyz.append("Z")
            continue
        mean_s = np.mean(sales)
        std_s = np.std(sales, ddof=1)
        cv = std_s / mean_s if mean_s > 0 else float("inf")
        xyz.append("X" if cv < 0.5 else "Y" if cv < 1.0 else "Z")
    return xyz


def _to_date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
        return None
    try:
        ts = pd.to_datetime(v)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


# ─────────────────────────── order calculation ───────────────────────────

def calculate_orders(data, lead_time_months, target_stock_months, service_level,
                     min_order_qty, rounding, sales_months_to_use, excluded_months=None,
                     target_fast_months=None, target_slow_months=None, today=None):
    """excluded_months: 0-based indices into the 12-month array (0 = most recent).
    target_fast_months / target_slow_months override target_stock_months for
    Rápido / Lento items; None means "same as target_stock_months"."""
    z_score = SERVICE_LEVEL_Z.get(service_level, 1.65)
    excluded_set = set(excluded_months or [])
    today = today or dt.date.today()
    target_fast = target_stock_months if target_fast_months is None else float(target_fast_months)
    target_slow = target_stock_months if target_slow_months is None else float(target_slow_months)

    rows_list = [row for _, row in data.iterrows()]
    all_monthly_sales = [[float(row.get(c, 0)) for c in MONTH_COLS] for row in rows_list]

    abc_classes = calculate_abc([r.get("FOB", 0) for r in rows_list],
                                [r.get("Ventas UN", 0) for r in rows_list])
    xyz_classes = calculate_xyz(all_monthly_sales)

    # ── pass 1: forecast per item ──
    pre = []
    for idx, row in enumerate(rows_list):
        monthly_sales = all_monthly_sales[idx]
        recent_sales = [s for i, s in enumerate(monthly_sales[:sales_months_to_use]) if i not in excluded_set]
        all_12_filtered = [s for i, s in enumerate(monthly_sales) if i not in excluded_set]

        demand_pattern, adi, cv2 = classify_demand(all_12_filtered)
        non_zero_recent = [s for s in recent_sales if s > 0]

        if demand_pattern == "Sin Demanda":
            avg = 0.0
            method = "N/A"
        elif demand_pattern in ("Intermitente", "Irregular"):
            avg = croston_forecast(all_12_filtered)
            method = "Croston SBA"
        else:
            avg = moving_average(recent_sales)
            if avg == 0:
                avg = float(row.get("Vta Prom Mensual", 0))
            method = "Promedio Móvil"
        pre.append((recent_sales, non_zero_recent, demand_pattern, adi, cv2, avg, method))

    velocity = classify_velocity([p[5] for p in pre])

    # ── pass 2: order quantities ──
    results = []
    for idx, row in enumerate(rows_list):
        monthly_sales = all_monthly_sales[idx]
        recent_sales, non_zero_recent, demand_pattern, adi, cv2, avg_monthly_sales, forecast_method = pre[idx]
        vel = velocity[idx]
        target_used = target_fast if vel == "Rápido" else target_slow if vel == "Lento" else target_stock_months

        stock_total = float(row.get("Stock Total", 0))
        stock_cedi = float(row.get("Stock CEDI", 0))
        stock_tiendas = float(row.get("Stock Tiendas", 0))

        # dates: last purchase / last delivery → observed lead time, in-transit qty
        fecha_compra_d = _to_date(row.get("Fe. Comp"))
        fecha_entrega_d = _to_date(row.get("Fecha Ult. Entrega"))
        cant_compra = int(row.get("Cant.", 0))
        lead_time_obs_days = None
        if fecha_compra_d and fecha_entrega_d and fecha_entrega_d >= fecha_compra_d:
            lead_time_obs_days = (fecha_entrega_d - fecha_compra_d).days
        en_transito = cant_compra if (fecha_entrega_d and fecha_entrega_d > today) else 0
        stock_posicion = stock_total + en_transito

        if avg_monthly_sales > 0 and len(recent_sales) >= 2:
            std_demand = float(np.std(recent_sales, ddof=1))
            safety_stock_units = z_score * std_demand * np.sqrt(lead_time_months)
        else:
            std_demand = 0.0
            safety_stock_units = 0.0

        lead_time_demand = avg_monthly_sales * lead_time_months
        rop = lead_time_demand + safety_stock_units
        lead_time_met = stock_posicion >= lead_time_demand or avg_monthly_sales == 0

        if lead_time_met:
            # stock survives the lead time: order up to (LT + target) coverage
            demand_coverage = avg_monthly_sales * (lead_time_months + target_used)
            suggested_qty = max(0.0, demand_coverage + safety_stock_units - stock_posicion)
        else:
            # stock runs out before arrival: the order alone must cover the target
            demand_coverage = avg_monthly_sales * target_used
            suggested_qty = max(0.0, demand_coverage + safety_stock_units)

        if 0 < suggested_qty < min_order_qty:
            suggested_qty = min_order_qty
        if rounding > 1 and suggested_qty > 0:
            suggested_qty = int(np.ceil(suggested_qty / rounding) * rounding)
        else:
            suggested_qty = int(np.ceil(suggested_qty))

        months_of_stock = (stock_posicion / avg_monthly_sales) if avg_monthly_sales > 0 else float("inf")

        if avg_monthly_sales == 0:
            status = "Sin Ventas"
        elif stock_posicion <= rop * 0.5:
            status = "Crítico"
        elif stock_posicion <= rop:
            status = "Bajo"
        elif suggested_qty > 0:
            status = "Reponer"
        else:
            status = "OK"

        fob = float(row.get("FOB", 0))
        fecha_compra = f"{MESES_ES[fecha_compra_d.month - 1]} {fecha_compra_d.year}" if fecha_compra_d else "-"
        fecha_entrega = f"{MESES_ES[fecha_entrega_d.month - 1]} {fecha_entrega_d.year}" if fecha_entrega_d else "-"

        results.append({
            "material": str(row.get("Material", "")),
            "marca": str(row.get("Marca", "")),
            "cod_proveedor": str(row.get("Cod. Proveedor Actual", "")),
            "art_proveedor": str(row.get("No. Art. Proveedor", "")),
            "descripcion": str(row.get("Descripcion", "")),
            "grupo": str(row.get("Grupo", "")),
            "cod_barra": str(row.get("Cod. Barra", "")),
            "abc_proveedor": str(row.get("ABC Proveedor", "")),
            "fecha_compra": fecha_compra,
            "fecha_compra_year": fecha_compra_d.year if fecha_compra_d else None,
            "fecha_entrega": fecha_entrega,
            "cant_compra": cant_compra,
            "lead_time_obs_days": lead_time_obs_days,
            "en_transito": int(en_transito),
            "abc": abc_classes[idx],
            "xyz": xyz_classes[idx],
            "abc_xyz": f"{abc_classes[idx]}{xyz_classes[idx]}",
            "demand_pattern": demand_pattern,
            "adi": adi,
            "cv2": cv2,
            "forecast_method": forecast_method,
            "velocidad": vel,
            "target_months_used": target_used,
            "stock_cedi": int(stock_cedi),
            "stock_tiendas": int(stock_tiendas),
            "stock_total": int(stock_total),
            "stock_posicion": int(stock_posicion),
            "fob": round(fob, 2),
            "costo": round(float(row.get("Costo", 0)), 2),
            "pvp": round(float(row.get("PVP", 0)), 2),
            "avg_monthly_sales": round(avg_monthly_sales, 2),
            "active_months": len(non_zero_recent),
            "total_months_analyzed": len(recent_sales),
            "std_demand": round(std_demand, 2),
            "total_sales": int(row.get("Ventas UN", 0)),
            "ventas_mes_actual": int(row.get(CURRENT_MONTH_COL, 0)),
            "months_of_stock": round(months_of_stock, 1) if months_of_stock != float("inf") else "∞",
            "safety_stock_units": round(safety_stock_units, 1),
            "rop": round(rop, 1),
            "demand_coverage": round(demand_coverage, 1),
            "lead_time_met": bool(lead_time_met),
            "suggested_qty": int(suggested_qty),
            "order_value_fob": round(suggested_qty * fob, 2),
            "status": status,
            "monthly_sales_all": monthly_sales,
            "monthly_sales": monthly_sales[:sales_months_to_use],
        })
    return results


def build_summary(results):
    abc_counts = {}
    demand_pattern_counts = {}
    velocity_counts = {}
    for r in results:
        abc_counts[r["abc_xyz"]] = abc_counts.get(r["abc_xyz"], 0) + 1
        demand_pattern_counts[r["demand_pattern"]] = demand_pattern_counts.get(r["demand_pattern"], 0) + 1
        velocity_counts[r["velocidad"]] = velocity_counts.get(r["velocidad"], 0) + 1
    lt_days = [r["lead_time_obs_days"] for r in results if r["lead_time_obs_days"] is not None]
    lt_median = float(np.median(lt_days)) if lt_days else None
    return {
        "total_items": len(results),
        "items_to_order": sum(1 for r in results if r["suggested_qty"] > 0),
        "total_order_value_fob": round(sum(r["order_value_fob"] for r in results), 2),
        "total_units": sum(r["suggested_qty"] for r in results),
        "critical_items": sum(1 for r in results if r["status"] == "Crítico"),
        "low_items": sum(1 for r in results if r["status"] == "Bajo"),
        "no_sales_items": sum(1 for r in results if r["status"] == "Sin Ventas"),
        "in_transit_items": sum(1 for r in results if r["en_transito"] > 0),
        "lead_time_obs_median_days": lt_median,
        "lead_time_obs_median_months": round(lt_median / 30.4, 1) if lt_median is not None else None,
        "lead_time_obs_items": len(lt_days),
        "abc_xyz_counts": abc_counts,
        "demand_pattern_counts": demand_pattern_counts,
        "velocity_counts": velocity_counts,
    }


def _opt_float(v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    return float(v)


def params_from_fields(fields):
    """Build calculate_orders kwargs from form fields (strings)."""
    excluded_str = fields.get("excluded_months", "") or ""
    excluded_months = [int(x) for x in excluded_str.split(",") if x.strip().isdigit()]
    return {
        "lead_time_months": float(fields.get("lead_time_months", "3") or 3),
        "target_stock_months": float(fields.get("target_stock_months", "3") or 3),
        "service_level": int(fields.get("service_level", "95") or 95),
        "min_order_qty": int(fields.get("min_order_qty", "1") or 1),
        "rounding": int(fields.get("rounding", "1") or 1),
        "sales_months_to_use": int(fields.get("sales_months_to_use", "3") or 3),
        "excluded_months": excluded_months,
        "target_fast_months": _opt_float(fields.get("target_fast_months")),
        "target_slow_months": _opt_float(fields.get("target_slow_months")),
    }


def process_upload(file_bytes, fields):
    """Shared entry point: parse workbook bytes, compute, and return the response dict."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        data = parse_excel(tmp_path)
    finally:
        os.unlink(tmp_path)

    params = params_from_fields(fields)
    results = calculate_orders(data, **params)
    return {
        "results": results,
        "summary": build_summary(results),
        "params": params,
    }


# ─────────────────────────── vercel handler ───────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json_response(400, {"error": "Se requiere multipart/form-data"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_FILE_SIZE:
            self._json_response(400, {"error": f"El archivo excede el límite de {MAX_FILE_SIZE // (1024*1024)}MB"})
            return

        try:
            body = self.rfile.read(content_length)
            fields, file_data, file_name = parse_multipart(content_type, body)
        except Exception as e:
            self._json_response(400, {"error": f"Error al procesar los datos del formulario: {str(e)}"})
            return

        if file_data is None:
            self._json_response(400, {"error": "No se subió ningún archivo"})
            return

        if not file_name or not file_name.lower().endswith((".xlsx", ".xls")):
            self._json_response(400, {"error": "Por favor suba un archivo Excel (.xlsx o .xls)"})
            return

        try:
            self._json_response(200, process_upload(file_data, fields))
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
        except Exception as e:
            self._json_response(500, {"error": f"Error interno del servidor: {str(e)}"})

    def _json_response(self, status_code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
