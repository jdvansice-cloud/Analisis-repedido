import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder="public")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

MONTH_COLS = [f"Vta {str(i).zfill(2)}" for i in range(1, 13)]

# Z-scores for common service levels
SERVICE_LEVEL_Z = {
    85: 1.04,
    90: 1.28,
    92: 1.41,
    95: 1.65,
    97: 1.88,
    98: 2.05,
    99: 2.33,
}


def parse_excel(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0, header=None)
    except Exception as e:
        raise ValueError(f"Error al leer el archivo Excel: {str(e)}")

    if len(df) == 0:
        raise ValueError("El archivo Excel está vacío")

    header_row = None
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str).str.strip().tolist()
        if "Material" in row_vals:
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            "No se encontró la fila de encabezado con columna 'Material'. "
            "Asegúrese de que su archivo tenga una columna llamada 'Material' en las primeras 10 filas."
        )

    headers = df.iloc[header_row].astype(str).str.strip().tolist()
    data = df.iloc[header_row + 1:].copy()
    data.columns = headers

    data = data[data["Material"].notna() & (data["Material"].astype(str).str.strip() != "")]
    data = data[data["Material"].astype(str).str.lower() != "nan"]

    if len(data) == 0:
        raise ValueError("No se encontraron datos después de la fila de encabezado")

    numeric_cols = [
        "Cant.", "Stock CEDI", "K001 / Q001", "Stock Tiendas", "Stock Total",
        "FOB", "Costo", "PVP", "Vta Prom Mensual", "Ventas UN",
    ] + MONTH_COLS

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    return data


def classify_demand(monthly_sales):
    """Classify demand pattern using ADI (Average Demand Interval) and CV² (Coefficient of Variation squared).

    Categories (Syntetos-Boylan framework):
    - Suave (Smooth): ADI < 1.32, CV² < 0.49 → regular demand, use moving average
    - Err\u00e1tica (Erratic): ADI < 1.32, CV² >= 0.49 → frequent but variable, use moving average with higher safety
    - Intermitente (Intermittent): ADI >= 1.32, CV² < 0.49 → sporadic but consistent size, use Croston
    - Irregular (Lumpy): ADI >= 1.32, CV² >= 0.49 → sporadic and variable, use Croston/SBA
    """
    non_zero = [s for s in monthly_sales if s > 0]

    if len(non_zero) == 0:
        return "Sin Demanda", 0, 0

    # ADI: average number of periods between demands
    n_periods = len(monthly_sales)
    n_demands = len(non_zero)
    adi = n_periods / n_demands

    # CV²: squared coefficient of variation of non-zero demands
    if len(non_zero) >= 2:
        mean_nz = np.mean(non_zero)
        std_nz = np.std(non_zero, ddof=1)
        cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0
    else:
        cv2 = 0

    # Syntetos-Boylan classification thresholds
    if adi < 1.32 and cv2 < 0.49:
        return "Suave", round(adi, 2), round(cv2, 2)
    elif adi < 1.32 and cv2 >= 0.49:
        return "Err\u00e1tica", round(adi, 2), round(cv2, 2)
    elif adi >= 1.32 and cv2 < 0.49:
        return "Intermitente", round(adi, 2), round(cv2, 2)
    else:
        return "Irregular", round(adi, 2), round(cv2, 2)


def croston_forecast(monthly_sales, alpha=0.15):
    """Croston's method with SBA (Syntetos-Boylan Approximation) bias correction.
    Better for intermittent/lumpy demand than simple averages.
    Returns forecast per period.
    """
    non_zero = [s for s in monthly_sales if s > 0]
    if len(non_zero) == 0:
        return 0.0

    if len(non_zero) == 1:
        # Only one demand occurrence — use it divided by total periods
        return non_zero[0] / len(monthly_sales)

    # Initialize with first non-zero demand
    z_hat = non_zero[0]  # demand size estimate
    p_hat = 1.0  # inter-demand interval estimate

    q = 0  # periods since last demand
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

    # SBA bias correction factor (1 - alpha/2)
    forecast = (z_hat / p_hat) * (1 - alpha / 2)
    return max(0, forecast)


def calculate_abc(items_data):
    """ABC classification based on annual sales value (FOB × total sales)."""
    values = []
    for row in items_data:
        fob = float(row.get("FOB", 0))
        sales = float(row.get("Ventas UN", 0))
        values.append(fob * sales)

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
        else:
            abc[idx] = "C"

    return abc


def calculate_xyz(items_sales):
    """XYZ classification based on demand variability (CV of monthly sales).
    X: CV < 0.5 (stable demand)
    Y: 0.5 <= CV < 1.0 (variable demand)
    Z: CV >= 1.0 (highly variable / unpredictable)
    """
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
    """excluded_months: list of 0-based indices into monthly_sales array to exclude.
    Index 0 = Vta 01 (most recent), index 11 = Vta 12 (oldest)."""
    z_score = SERVICE_LEVEL_Z.get(service_level, 1.65)
    if excluded_months is None:
        excluded_months = []
    excluded_set = set(excluded_months)

    # Collect all monthly sales for ABC/XYZ classification
    all_monthly_sales = []
    rows_list = list(data.iterrows())

    for _, row in rows_list:
        monthly_sales = []
        for i in range(1, 13):
            col = f"Vta {str(i).zfill(2)}"
            if col in data.columns:
                monthly_sales.append(float(row.get(col, 0)))
        all_monthly_sales.append(monthly_sales)

    # ABC classification (uses all data regardless of exclusions)
    abc_classes = calculate_abc([row for _, row in rows_list])
    xyz_classes = calculate_xyz(all_monthly_sales)

    results = []
    for idx, (_, row) in enumerate(rows_list):
        monthly_sales = all_monthly_sales[idx]

        # Build filtered sales arrays (replace excluded months with None marker)
        recent_sales_raw = monthly_sales[:sales_months_to_use]
        recent_sales = [s for i, s in enumerate(recent_sales_raw) if i not in excluded_set]
        all_12_filtered = [s for i, s in enumerate(monthly_sales) if i not in excluded_set]

        # --- Demand Classification (uses filtered 12 months) ---
        demand_pattern, adi, cv2 = classify_demand(all_12_filtered)

        # --- Forecast based on demand pattern ---
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

        # --- Statistical Safety Stock ---
        if avg_monthly_sales > 0 and len(recent_sales) >= 2:
            std_demand = np.std(recent_sales, ddof=1)
            safety_stock_units = z_score * std_demand * np.sqrt(lead_time_months)
        else:
            safety_stock_units = 0

        # --- Reorder Point (ROP) ---
        rop = (avg_monthly_sales * lead_time_months) + safety_stock_units

        # --- Demand during target coverage period ---
        demand_coverage = avg_monthly_sales * (lead_time_months + target_stock_months)

        # --- Suggested Order Qty ---
        suggested_qty = demand_coverage + safety_stock_units - stock_total
        suggested_qty = max(0, suggested_qty)

        if 0 < suggested_qty < min_order_qty:
            suggested_qty = min_order_qty

        if rounding > 1 and suggested_qty > 0:
            suggested_qty = int(np.ceil(suggested_qty / rounding) * rounding)
        else:
            suggested_qty = int(np.ceil(suggested_qty))

        # --- Months of Stock ---
        months_of_stock = (stock_total / avg_monthly_sales) if avg_monthly_sales > 0 else float("inf")

        # --- Status ---
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
        order_value = suggested_qty * fob

        # Last purchase
        MESES_ES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        fecha_compra_raw = row.get("Fe. Comp", "")
        fecha_compra = "-"
        fecha_compra_year = None
        if pd.notna(fecha_compra_raw):
            try:
                dt = pd.to_datetime(fecha_compra_raw)
                fecha_compra = f"{MESES_ES[dt.month - 1]} {dt.year}"
                fecha_compra_year = dt.year
            except Exception:
                fecha_compra = str(fecha_compra_raw)
        cant_compra = int(row.get("Cant.", 0))

        # ABC-XYZ
        abc_class = abc_classes[idx]
        xyz_class = xyz_classes[idx]

        results.append({
            "material": str(row.get("Material", "")),
            "marca": str(row.get("Marca", "")),
            "cod_proveedor": str(row.get("Cod. Proveedor Actual", "")),
            "descripcion": str(row.get("Descripcion", "")),
            "fecha_compra": fecha_compra,
            "fecha_compra_year": fecha_compra_year,
            "cant_compra": cant_compra,
            "abc": abc_class,
            "xyz": xyz_class,
            "abc_xyz": f"{abc_class}{xyz_class}",
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
            "order_value_fob": round(order_value, 2),
            "status": status,
            "monthly_sales_all": monthly_sales,
            "monthly_sales": monthly_sales[:sales_months_to_use],
        })

    return results


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No se subi\u00f3 ning\u00fan archivo"}), 400

    file = request.files["file"]
    if not file.filename.endswith((".xlsx", ".xls")):
        return jsonify({"error": "Por favor suba un archivo Excel (.xlsx o .xls)"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            file.save(tmp.name)
            data = parse_excel(tmp.name)
            os.unlink(tmp.name)

        # Parse excluded months (comma-separated 0-based indices)
        excluded_str = request.form.get("excluded_months", "")
        excluded_months = []
        if excluded_str:
            excluded_months = [int(x) for x in excluded_str.split(",") if x.strip().isdigit()]

        params = {
            "lead_time_months": float(request.form.get("lead_time_months", 3)),
            "target_stock_months": float(request.form.get("target_stock_months", 3)),
            "service_level": int(request.form.get("service_level", 95)),
            "min_order_qty": int(request.form.get("min_order_qty", 1)),
            "rounding": int(request.form.get("rounding", 1)),
            "sales_months_to_use": int(request.form.get("sales_months_to_use", 3)),
            "excluded_months": excluded_months,
        }

        results = calculate_orders(data, **params)

        total_items = len(results)
        items_to_order = sum(1 for r in results if r["suggested_qty"] > 0)
        total_order_value = sum(r["order_value_fob"] for r in results)
        critical_items = sum(1 for r in results if r["status"] == "Cr\u00edtico")
        low_items = sum(1 for r in results if r["status"] == "Bajo")
        no_sales_items = sum(1 for r in results if r["status"] == "Sin Ventas")
        total_units = sum(r["suggested_qty"] for r in results)

        # ABC-XYZ summary
        abc_counts = {}
        for r in results:
            key = r["abc_xyz"]
            abc_counts[key] = abc_counts.get(key, 0) + 1

        demand_pattern_counts = {}
        for r in results:
            key = r["demand_pattern"]
            demand_pattern_counts[key] = demand_pattern_counts.get(key, 0) + 1

        return jsonify({
            "results": results,
            "summary": {
                "total_items": total_items,
                "items_to_order": items_to_order,
                "total_order_value_fob": round(total_order_value, 2),
                "total_units": total_units,
                "critical_items": critical_items,
                "low_items": low_items,
                "no_sales_items": no_sales_items,
                "abc_xyz_counts": abc_counts,
                "demand_pattern_counts": demand_pattern_counts,
            },
            "params": params,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs("public", exist_ok=True)
    app.run(debug=True, port=5050)
