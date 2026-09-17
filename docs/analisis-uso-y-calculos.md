# Análisis de uso y cálculo de pedidos

Fecha: 2026-09-17. Archivo de referencia: `ARCA - pedido.xlsx` (69 SKUs, marca ARCA, grupo AD03, un proveedor).

## 1. Cómo se usa la app (flujo previsto)

1. El comprador exporta de Arrocha el reporte de repedido con los artículos que quiere analizar (normalmente una marca, categoría o proveedor).
2. Sube ese único archivo, ajusta los parámetros (lead time, cobertura objetivo, historial) y calcula.
3. Revisa la tabla: estado, patrón de demanda, velocidad, cantidad sugerida. Edita cantidades a mano donde tenga información que el modelo no tiene (promociones, descontinuados, mínimos del proveedor).
4. Exporta a Excel o CSV para armar la orden de compra. El código de artículo del proveedor (`No. Art. Proveedor`) ahora viaja en la exportación.

Un archivo por vez es la unidad de trabajo correcta: las clasificaciones ABC y de velocidad son relativas al conjunto subido, así que mezclar categorías con ritmos muy distintos distorsiona las clases.

## 2. Formato nuevo del archivo

| Columna nueva | Equivalente anterior | Notas |
|---|---|---|
| `N° Material` | `Material` | |
| `Mes12` … `Mes1` | `Vta 12` … `Vta 01` | `Mes1` = último mes completo. Verificado con la columna `Actual` adyacente y con artículos nuevos cuya venta arranca en `Mes1`. |
| `Actual` | (no existía) | Mes en curso, parcial. Se muestra, no se pronostica. |
| `Fecha Ult. Compra` / `Fecha Ult. Entrega` | `Fe. Comp` / (no existía) | La diferencia es el lead time real observado. |
| `Ult. Cantidad` | `Cant.` | Si la entrega es futura, se trata como en tránsito. |
| `PV` | `PVP` | |
| `Vta Prom Mes` / `Ventas UN Total` | `Vta Prom Mensual` / `Ventas UN` | Consistentes con la suma de los 12 meses. |
| `Cod. Proveedor` | `Cod. Proveedor Actual` | |
| `No. Art. Proveedor`, `ABC`, `Grupo`, `Cod. Barra`, `Tecla` | (no existían) | Se conservan y exportan. |

Cinco filas (gift sets) traen ventas en blanco. Se interpretan como cero.

## 3. Qué dice el archivo sobre el negocio

| Indicador | Valor |
|---|---|
| Lead time observado (mediana compra → entrega) | 29 días, rango 22 a 48 |
| Rotación media | 15 un/mes por SKU, mediana 12, máximo 42 |
| Meses de stock actuales (mediana) | 7.3 meses, 25 % de los SKUs por encima de 11.7 |
| Ventas últimos 3 meses vs promedio 12 meses | ratio mediano 1.08 (demanda estable, ligera alza) |
| Patrón de demanda | 38 Suave, 15 Errática, 9 Irregular, 2 Intermitente, 5 sin demanda |

Conclusiones prácticas:

- **El lead time real es de un mes, no tres.** Con el parámetro en 3 meses el cálculo asume que el stock debe aguantar tres veces más de lo que tarda el proveedor. Si el comprador ordena cada mes, el parámetro correcto está cerca de 1 a 1.5 meses. Si ordena cada trimestre, 3 sigue siendo razonable porque el parámetro debe cubrir hasta el próximo pedido. La app ahora muestra el lead time observado y avisa cuando difiere del parámetro en más de un mes.
- **Hay sobre-stock en la cola lenta.** Un cuarto de los SKUs tiene más de un año de inventario. Con una cobertura uniforme de 6 meses esos artículos igual reciben pedido en cuanto bajan del umbral. La cobertura diferenciada por velocidad ataca exactamente eso.
- **Las velas y aceites de 100 ml concentran el volumen.** 16 SKUs "Rápido" explican la mitad de las unidades pronosticadas; 24 SKUs "Lento" suman menos del 15 %.

## 4. Revisión de los cálculos

### 4.1 Lo que estaba bien y se mantiene

- **Clasificación Syntetos-Boylan** (ADI y CV²) para elegir método de pronóstico, con Croston SBA para demanda intermitente. Es el enfoque estándar y funciona bien con las series de 12 meses.
- **Stock de seguridad** Z × σ × √LT. Correcto para demanda mensual con lead time en meses.
- **Cantidad sugerida por casos (LT OK / LT NO).** Aunque parezca una excepción, es la política order-up-to bien planteada: si el stock alcanza hasta la llegada, se pide la diferencia; si no alcanza, el stock se agota antes de recibir y el pedido debe cubrir por sí solo la cobertura objetivo. No cambia.
- **ABC por valor y XYZ por variabilidad.** Correctos.

### 4.2 Lo que se corrigió

| Problema | Efecto que tenía | Cambio |
|---|---|---|
| Promedio móvil dividía por meses con venta > 0 | Inflaba el pronóstico de artículos lentos: 10, 0, 10 daba 10 un/mes en vez de 6.7. Con historial de 12 meses y patrón Suave el sesgo era pequeño, pero con 3 meses y artículos erráticos llegaba al 50 %. | Se divide por los meses transcurridos desde la primera venta del período. Los meses anteriores al lanzamiento de un artículo nuevo se ignoran; los ceros posteriores cuentan. |
| No se consideraban unidades en tránsito | Si la última orden aún no llegaba, el modelo la volvía a pedir. | Si `Fecha Ult. Entrega` es futura, `Ult. Cantidad` se suma al stock de cálculo (columna Tránsito). En el archivo de hoy no hay ninguna, pero aparecerá en cuanto se analice justo después de ordenar. |
| Lead time solo como parámetro manual | Sin referencia real para elegirlo. | Se calcula por SKU y se muestra la mediana en el resumen. |
| Una sola cobertura objetivo para todo | Los lentos acumulaban inventario al mismo ritmo que los rápidos. | Velocidad Rápido / Medio / Lento y parámetros de cobertura separados para Rápido y Lento. |
| Croston recorría la serie del mes más reciente al más antiguo | El suavizado terminaba ponderando más el pasado lejano. | Se invierte la serie para que el suavizado termine en el mes más reciente. |
| Lógica duplicada en `app.py` y `api/upload.py` | Ya habían divergido (las columnas costo/PVP existían en uno y no en el otro). | Toda la lógica vive en `api/upload.py`; `app.py` la importa. |

### 4.3 Impacto sobre el archivo de ejemplo

Parámetros: lead time 1 mes, nivel de servicio 95 %, historial 12 meses.

| Escenario | Unidades a pedir | Valor FOB |
|---|---|---|
| Cobertura 6 meses uniforme | 3 145 | $5 512 |
| Rápidos 6, Medios 6, Lentos 3 | 2 985 | $5 119 |
| Parámetros antiguos (LT 3, objetivo 3) | 2 743 | $4 839 |

El ahorro de la cobertura diferenciada (160 unidades, unos $400) es pequeño en valor porque los lentos son baratos, pero evita inmovilizar inventario que ya tiene más de un año de cobertura.

## 5. Recomendaciones de uso

1. **Lead time**: usar 1 a 1.5 meses si se ordena mensualmente a este proveedor. Revisar el valor observado en el resumen cada vez.
2. **Historial**: 12 meses para candles y aromas (demanda estable con estacionalidad navideña visible en Mes9 y Mes10). Excluir meses atípicos con los chips de exclusión en lugar de acortar el historial.
3. **Cobertura**: 6 meses para Rápidos, 3 para Lentos como punto de partida. Guardarlo como preset.
4. **Lentos con más de 12 meses de stock**: filtrar por Velocidad = Lento, ordenar por Meses Stock y decidir a mano si se descontinúan. El modelo no pedirá, pero tampoco sugiere liquidar.
5. **Artículos nuevos** (venta solo en los últimos 2 o 3 meses): el promedio móvil ya ignora los meses previos al lanzamiento, pero el stock de seguridad usa la desviación de todo el período, así que puede quedar alto. Revisar a mano hasta tener 6 meses de historia.

## 6. Pendientes sugeridos

- Estacionalidad explícita: el pico de Mes9 y Mes10 (diciembre y noviembre) se diluye en el promedio. Un factor estacional por mes objetivo mejoraría el pedido de octubre.
- Mínimos y múltiplos por proveedor guardados en preset, en lugar de un valor global.
- Marcar automáticamente candidatos a descontinuar (Lento con más de 12 meses de stock).
