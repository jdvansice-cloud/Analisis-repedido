import datetime as dt
import os
import sys
import tempfile
import unittest

import openpyxl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"))

from upload import (  # noqa: E402
    parse_excel,
    calculate_orders,
    build_summary,
    moving_average,
    classify_velocity,
    MONTH_COLS,
)

TODAY = dt.date(2026, 9, 17)

NEW_HEADERS = [
    "N° Material", "No. Art. Proveedor", "Descripcion", "Marca", "ABC", "Grupo",
    "Cod. Barra", "Tecla", "Cod. Proveedor", "Fecha Ult. Compra", "Fecha Ult. Entrega",
    "Ult. Cantidad", "Stock CEDI", "Stock Tiendas", "Stock Total", "FOB", "Costo", "PV",
    "Vta Prom Mes", "Ventas UN Total",
    "Mes12", "Mes11", "Mes10", "Mes9", "Mes8", "Mes7", "Mes6", "Mes5", "Mes4", "Mes3", "Mes2", "Mes1",
    "Actual",
]

OLD_HEADERS = [
    "Material", "Marca", "Cod. Proveedor Actual", "Descripcion", "Fe. Comp", "Cant.",
    "Stock CEDI", "Stock Tiendas", "Stock Total", "FOB", "Costo", "PVP",
    "Vta Prom Mensual", "Ventas UN",
] + MONTH_COLS


def write_xlsx(headers, rows, title_rows=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    for tr in title_rows or []:
        ws.append(tr)
    ws.append(headers)
    for r in rows:
        ws.append(r)
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    wb.save(path)
    return path


def new_row(material="100673483", mes=None, actual=28, stock=95, compra=dt.datetime(2026, 6, 15),
            entrega=dt.datetime(2026, 7, 14), ult_cant=144, fob=1.25):
    # mes given as [Mes1 (most recent) ... Mes12 (oldest)]
    mes = mes or [57, 15, 29, 43, 32, 26, 14, 5, 47, 32, 47, 42]
    mes12_to_1 = list(reversed(mes))
    total = sum(mes)
    return [material, "0841-MA780A", "AROMA CANDLE", "ARCA", "A", "AD03", 2940001285274, 29,
            "0050002174", compra, entrega, ult_cant, 0, stock, stock, fob, 1.73, 5.5,
            total / 12, total] + mes12_to_1 + [actual]


DEFAULT_PARAMS = dict(
    lead_time_months=1, target_stock_months=3, service_level=95, min_order_qty=1,
    rounding=1, sales_months_to_use=12, excluded_months=None, today=TODAY,
)


class ParseNewFormat(unittest.TestCase):
    def test_maps_new_headers_to_canonical(self):
        path = write_xlsx(NEW_HEADERS, [new_row()])
        data = parse_excel(path)
        os.unlink(path)
        for col in ["Material", "Descripcion", "Stock Total", "FOB", "PVP", "Ventas UN",
                    "Vta Prom Mensual", "Fe. Comp", "Cant.", "Cod. Proveedor Actual"] + MONTH_COLS:
            self.assertIn(col, data.columns, col)
        row = data.iloc[0]
        self.assertEqual(str(row["Material"]), "100673483")
        self.assertEqual(row["PVP"], 5.5)
        self.assertEqual(row["Ventas UN"], 389)

    def test_mes1_is_most_recent(self):
        path = write_xlsx(NEW_HEADERS, [new_row()])
        data = parse_excel(path)
        os.unlink(path)
        row = data.iloc[0]
        self.assertEqual(row["Vta 01"], 57)   # Mes1
        self.assertEqual(row["Vta 12"], 42)   # Mes12
        self.assertEqual(row["Vta Actual"], 28)

    def test_extra_columns_kept(self):
        path = write_xlsx(NEW_HEADERS, [new_row()])
        data = parse_excel(path)
        os.unlink(path)
        row = data.iloc[0]
        self.assertEqual(row["No. Art. Proveedor"], "0841-MA780A")
        self.assertEqual(row["ABC Proveedor"], "A")
        self.assertEqual(str(row["Cod. Barra"]), "2940001285274")

    def test_blank_sales_rows_become_zero(self):
        r = new_row(material="100673513")
        r[18:] = [None] * (len(r) - 18)
        path = write_xlsx(NEW_HEADERS, [r])
        data = parse_excel(path)
        os.unlink(path)
        self.assertEqual(data.iloc[0]["Vta 01"], 0)
        self.assertEqual(data.iloc[0]["Ventas UN"], 0)

    def test_header_row_not_first(self):
        path = write_xlsx(NEW_HEADERS, [new_row()], title_rows=[["Reporte de pedido"], []])
        data = parse_excel(path)
        os.unlink(path)
        self.assertEqual(len(data), 1)

    def test_material_float_becomes_clean_string(self):
        r = new_row()
        r[0] = 100673483.0
        path = write_xlsx(NEW_HEADERS, [r])
        data = parse_excel(path)
        os.unlink(path)
        self.assertEqual(data.iloc[0]["Material"], "100673483")


class ParseOldFormat(unittest.TestCase):
    def test_old_format_still_parses(self):
        row = ["A1", "M", "P1", "Desc", dt.datetime(2025, 3, 1), 10, 1, 2, 3, 2.0, 2.5, 5.0, 1.0, 12] + [1] * 12
        path = write_xlsx(OLD_HEADERS, [row])
        data = parse_excel(path)
        os.unlink(path)
        self.assertEqual(data.iloc[0]["Material"], "A1")
        self.assertEqual(data.iloc[0]["Vta 05"], 1)
        self.assertEqual(data.iloc[0]["Vta Actual"], 0)

    def test_missing_material_header_raises(self):
        path = write_xlsx(["Foo", "Bar"], [[1, 2]])
        with self.assertRaises(ValueError):
            parse_excel(path)
        os.unlink(path)


class MovingAverage(unittest.TestCase):
    def test_steady_demand(self):
        self.assertAlmostEqual(moving_average([10, 10, 10]), 10)

    def test_zero_month_counts_as_zero_demand(self):
        # 10, 0, 10 over 3 months -> 6.67 per month, not 10
        self.assertAlmostEqual(moving_average([10, 0, 10]), 20 / 3)

    def test_new_item_ignores_months_before_first_sale(self):
        # index 0 = most recent. Item started selling 2 months ago.
        self.assertAlmostEqual(moving_average([10, 6, 0, 0, 0, 0]), 8)

    def test_all_zero(self):
        self.assertEqual(moving_average([0, 0, 0]), 0)


class Velocity(unittest.TestCase):
    def test_pareto_tiers(self):
        rates = [40, 30, 10, 5, 3, 2, 0.5, 0]
        tiers = classify_velocity(rates)
        self.assertEqual(tiers[0], "Rápido")
        self.assertEqual(tiers[1], "Rápido")
        self.assertEqual(tiers[2], "Medio")
        self.assertEqual(tiers[6], "Lento")   # below 1 unit/month
        self.assertEqual(tiers[7], "-")       # no demand

    def test_all_equal_rates(self):
        tiers = classify_velocity([5, 5, 5, 5])
        self.assertEqual(len(tiers), 4)
        self.assertTrue(all(t in ("Rápido", "Medio", "Lento") for t in tiers))


class Orders(unittest.TestCase):
    def _run(self, rows, **overrides):
        path = write_xlsx(NEW_HEADERS, rows)
        data = parse_excel(path)
        os.unlink(path)
        params = dict(DEFAULT_PARAMS)
        params.update(overrides)
        return calculate_orders(data, **params)

    def test_observed_lead_time_days(self):
        res = self._run([new_row()])
        self.assertEqual(res[0]["lead_time_obs_days"], 29)

    def test_in_transit_added_to_stock_position(self):
        past = self._run([new_row(entrega=dt.datetime(2026, 7, 14))])[0]
        future = self._run([new_row(entrega=dt.datetime(2026, 10, 14))])[0]
        self.assertEqual(past["en_transito"], 0)
        self.assertEqual(future["en_transito"], 144)
        self.assertEqual(future["stock_posicion"], 95 + 144)
        self.assertLess(future["suggested_qty"], past["suggested_qty"])

    def test_order_up_to_when_stock_covers_lead_time(self):
        # steady 10/month, lead time 1, target 3, stock 20 -> LT met
        res = self._run([new_row(mes=[10] * 12, stock=20)], service_level=95)[0]
        self.assertTrue(res["lead_time_met"])
        self.assertEqual(res["safety_stock_units"], 0)   # zero variance
        self.assertEqual(res["suggested_qty"], 10 * (1 + 3) - 20)

    def test_no_deduction_when_stock_below_lead_time_demand(self):
        res = self._run([new_row(mes=[10] * 12, stock=5)])[0]
        self.assertFalse(res["lead_time_met"])
        self.assertEqual(res["suggested_qty"], 30)

    def test_velocity_specific_coverage(self):
        rows = [
            new_row(material="FAST", mes=[100] * 12, stock=0),
            new_row(material="SLOW", mes=[2] * 12, stock=0),
        ]
        res = {r["material"]: r for r in self._run(rows, target_stock_months=3,
                                                    target_fast_months=6, target_slow_months=1)}
        self.assertEqual(res["FAST"]["velocidad"], "Rápido")
        self.assertEqual(res["SLOW"]["velocidad"], "Lento")
        self.assertEqual(res["FAST"]["target_months_used"], 6)
        self.assertEqual(res["SLOW"]["target_months_used"], 1)
        self.assertEqual(res["FAST"]["suggested_qty"], 100 * 6)   # LT not met -> target only
        self.assertEqual(res["SLOW"]["suggested_qty"], 2 * 1)

    def test_current_month_passthrough(self):
        res = self._run([new_row(actual=28)])[0]
        self.assertEqual(res["ventas_mes_actual"], 28)

    def test_summary_has_lead_time_and_velocity(self):
        res = self._run([new_row(), new_row(material="X", mes=[1] * 12)])
        s = build_summary(res)
        self.assertEqual(s["lead_time_obs_median_days"], 29)
        self.assertIn("velocity_counts", s)


if __name__ == "__main__":
    unittest.main()
