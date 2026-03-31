import os
import json
import tempfile
from http.server import BaseHTTPRequestHandler
import cgi
import io
import pandas as pd
import numpy as np

MONTH_COLS = [f"Vta {str(i).zfill(2)}" for i in range(1, 13)]

SERVICE_LEVEL_Z = {
    85: 1.04, 90: 1.28, 92: 1.41, 95: 1.65, 97: 1.88, 98: 2.05, 99: 2.33,
}


def parse_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=0, header=None)
    header_row = None
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str).str.strip().tolist()
        if "Material" in row_vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError("No se encontr\u00f3 la fila de encabezado con columna 'Material'")

    headers = df.iloc[header_row].astype(str).str.strip().tolist()
    data = df.iloc[header_row + 1:].copy()
    data.columns = headers
    data = data[data["Material"].notna() & (data["Material"].astype(str).str.strip() != "")]
    data = data[data["Material"].astype(str).str.lower() != "nan"]

    numeric_cols = [
        "Cant.", "Stock CEDI", "K001 / Q001", "Stock Tiendas", "Stock Total",
        "FOB", "Costo", "PVP", "Vta Prom Mensual", "Ventas UN",
    ] + MONTH_COLS
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    return data


def classify_demand(monthly_sales):
    non_zero = [s for s in monthly_sales if s > 0]
    if len(non_zero) == 0:
        return "Sin Demanda", 0, 0
    n_periods = len(monthly_sales)
    n_demands = len(non_zero)
    adi = n_periods / n_demands
    if len(non_zero) >= 2:
        mean_nz = np.mean(non_zero)
        std_nz = np.std(non_zero, ddof=1)
        cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0
    else:
        cv2 = 0
    if adi < 1.32 and cv2 < 0.49:
        return "Suave", round(adi, 2), round(cv2, 2)
    elif adi < 1.32 and cv2 >= 0.49:
        return "Err\u00e1tica", round(adi, 2), round(cv2, 2)
    elif adi >= 1.32 and cv2 < 0.49:
        return "Intermitente", round(adi, 2), round(cv2, 2)
    else:
        return "Irregular", round(adi, 2), round(cv2, 2)


def croston_forecast(monthly_sales, alpha=0.15):
    non_zero = [s for s in monthly_sales if s > 0]
    if len(non_zero) == 0:
        return 0.0
    if len(non_zero) == 1:
        return non_zero[0] / len(monthly_sales)
    z_hat = non_zero[0]
    p_hat = 1.0
    q = 0
    first_demand_seen = False
    for val in monthly_sales:
        if val > 0:
            if first_demand_seen:
                q += 1
                z_hat = alpha * val + (1 - alpha) * z_hat
                p_hat = alpha * q + (1 - alpha) * p_hat
                q = 0
            else:
                z_hat = val
                first_demand_seen = True
                q = 0
        else:
            q += 1
    if p_hat <= 0:
        return 0.0
    forecast = (z_hat / p_hat) * (1 - alpha / 2)
    return max(0, forecast)


def calculate_abc(items_data):
    values = []
    for row in items_data:
        values.append(float(row.get("FOB", 0)) * float(row.get("Ventas UN", 0)))
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
        if cv < 0.5:
            xyz.append("X")
        elif cv < 1.0:
            xyz.append("Y")
        else:
            xyz.append("Z")
    return xyz


def calculate_orders(data, lead_time_months, target_stock_months, service_level,
                     min_order_qty, rounding, sales_months_to_use, excluded_months=None):
    z_score = SERVICE_LEVEL_Z.get(service_level, 1.65)
    if excluded_months is None:
        excluded_months = []
    excluded_set = set(excluded_months)

    all_monthly_sales = []
    rows_list = list(data.iterrows())
    for _, row in rows_list:
        monthly_sales = []
        for i in range(1, 13):
            col = f"Vta {str(i).zfill(2)}"
            if col in data.columns:
                monthly_sales.append(float(row.get(col, 0)))
        all_monthly_sales.append(monthly_sales)

    abc_classes = calculate_abc([row for _, row in rows_list])
    xyz_classes = calculate_xyz(all_monthly_sales)

    results = []
    for idx, (_, row) in enumerate(rows_list):
        monthly_sales = all_monthly_sales[idx]
        recent_sales_raw = monthly_sales[:sales_months_to_use]
        recent_sales = [s for i, s in enumerate(recent_sales_raw) if i not in excluded_set]
        all_12_filtered = [s for i, s in enumerate(monthly_sales) if i not in excluded_set]

        demand_pattern, adi, cv2 = classify_demand(all_12_filtered)
        non_zero_recent = [s for s in recent_sales if s > 0]

        if demand_pattern == "Sin Demanda":
            avg_monthly_sales = 0.0
            forecast_method = "N/A"
        elif demand_pattern in ("Intermitente", "Irregular"):
            avg_monthly_sales = croston_forecast(all_12_filtered)
            forecast_method = "Croston SBA"
        else:
            if non_zero_recent:
                avg_monthly_sales = sum(recent_sales) / len(non_zero_recent)
            else:
                avg_monthly_sales = float(row.get("Vta Prom Mensual", 0))
            forecast_method = "Promedio M\u00f3vil"

        active_months = len(non_zero_recent)
        total_months_analyzed = len(recent_sales)
        stock_total = float(row.get("Stock Total", 0))
        stock_cedi = float(row.get("Stock CEDI", 0))
        stock_tiendas = float(row.get("Stock Tiendas", 0))

        if avg_monthly_sales > 0 and len(recent_sales) >= 2:
            std_demand = np.std(recent_sales, ddof=1)
            safety_stock_units = z_score * std_demand * np.sqrt(lead_time_months)
        else:
            safety_stock_units = 0

        rop = (avg_monthly_sales * lead_time_months) + safety_stock_units
        demand_coverage = avg_monthly_sales * (lead_time_months + target_stock_months)
        suggested_qty = max(0, demand_coverage + safety_stock_units - stock_total)

        if 0 < suggested_qty < min_order_qty:
            suggested_qty = min_order_qty
        if rounding > 1 and suggested_qty > 0:
            suggested_qty = int(np.ceil(suggested_qty / rounding) * rounding)
        else:
            suggested_qty = int(np.ceil(suggested_qty))

        months_of_stock = (stock_total / avg_monthly_sales) if avg_monthly_sales > 0 else float("inf")

        if avg_monthly_sales == 0:
            status = "Sin Ventas"
        elif stock_total <= rop * 0.5:
            status = "Cr\u00edtico"
        elif stock_total <= rop:
            status = "Bajo"
        elif suggested_qty > 0:
            status = "Reponer"
        else:
            status = "OK"

        fob = float(row.get("FOB", 0))
        fecha_compra_raw = row.get("Fe. Comp", "")
        if pd.notna(fecha_compra_raw):
            try:
                fecha_compra = pd.to_datetime(fecha_compra_raw).strftime("%d/%m/%Y")
            except Exception:
                fecha_compra = str(fecha_compra_raw)
        else:
            fecha_compra = "-"

        results.append({
            "material": str(row.get("Material", "")),
            "marca": str(row.get("Marca", "")),
            "cod_proveedor": str(row.get("Cod. Proveedor Actual", "")),
            "descripcion": str(row.get("Descripcion", "")),
            "fecha_compra": fecha_compra,
            "cant_compra": int(row.get("Cant.", 0)),
            "abc": abc_classes[idx],
            "xyz": xyz_classes[idx],
            "abc_xyz": f"{abc_classes[idx]}{xyz_classes[idx]}",
            "demand_pattern": demand_pattern,
            "adi": adi,
            "cv2": cv2,
            "forecast_method": forecast_method,
            "stock_cedi": int(stock_cedi),
            "stock_tiendas": int(stock_tiendas),
            "stock_total": int(stock_total),
            "fob": round(fob, 2),
            "costo": round(float(row.get("Costo", 0)), 2),
            "pvp": round(float(row.get("PVP", 0)), 2),
            "avg_monthly_sales": round(avg_monthly_sales, 2),
            "active_months": active_months,
            "total_months_analyzed": total_months_analyzed,
            "std_demand": round(np.std(recent_sales, ddof=1), 2) if len(recent_sales) >= 2 else 0,
            "total_sales": int(row.get("Ventas UN", 0)),
            "months_of_stock": round(months_of_stock, 1) if months_of_stock != float("inf") else "\u221e",
            "safety_stock_units": round(safety_stock_units, 1),
            "rop": round(rop, 1),
            "demand_coverage": round(demand_coverage, 1),
            "suggested_qty": int(suggested_qty),
            "order_value_fob": round(suggested_qty * fob, 2),
            "status": status,
            "monthly_sales_all": monthly_sales,
            "monthly_sales": monthly_sales[:sales_months_to_use],
        })
    return results


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json_response(400, {"error": "Se requiere multipart/form-data"})
            return

        # Parse multipart form data
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)

        if "file" not in form:
            self._json_response(400, {"error": "No se subi\u00f3 ning\u00fan archivo"})
            return

        file_item = form["file"]
        if not file_item.filename or not file_item.filename.endswith((".xlsx", ".xls")):
            self._json_response(400, {"error": "Por favor suba un archivo Excel (.xlsx o .xls)"})
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp.write(file_item.file.read())
                tmp_path = tmp.name

            data = parse_excel(tmp_path)
            os.unlink(tmp_path)

            def get_field(name, default=""):
                if name in form:
                    return form[name].value
                return default

            excluded_str = get_field("excluded_months", "")
            excluded_months = []
            if excluded_str:
                excluded_months = [int(x) for x in excluded_str.split(",") if x.strip().isdigit()]

            params = {
                "lead_time_months": float(get_field("lead_time_months", "3")),
                "target_stock_months": float(get_field("target_stock_months", "3")),
                "service_level": int(get_field("service_level", "95")),
                "min_order_qty": int(get_field("min_order_qty", "1")),
                "rounding": int(get_field("rounding", "1")),
                "sales_months_to_use": int(get_field("sales_months_to_use", "3")),
                "excluded_months": excluded_months,
            }

            results = calculate_orders(data, **params)

            total_items = len(results)
            items_to_order = sum(1 for r in results if r["suggested_qty"] > 0)
            total_order_value = sum(r["order_value_fob"] for r in results)
            total_units = sum(r["suggested_qty"] for r in results)

            abc_counts = {}
            demand_pattern_counts = {}
            for r in results:
                abc_counts[r["abc_xyz"]] = abc_counts.get(r["abc_xyz"], 0) + 1
                demand_pattern_counts[r["demand_pattern"]] = demand_pattern_counts.get(r["demand_pattern"], 0) + 1

            self._json_response(200, {
                "results": results,
                "summary": {
                    "total_items": total_items,
                    "items_to_order": items_to_order,
                    "total_order_value_fob": round(total_order_value, 2),
                    "total_units": total_units,
                    "critical_items": sum(1 for r in results if r["status"] == "Cr\u00edtico"),
                    "low_items": sum(1 for r in results if r["status"] == "Bajo"),
                    "no_sales_items": sum(1 for r in results if r["status"] == "Sin Ventas"),
                    "abc_xyz_counts": abc_counts,
                    "demand_pattern_counts": demand_pattern_counts,
                },
                "params": params,
            })

        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, status_code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
