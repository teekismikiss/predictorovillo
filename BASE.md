### ROL
Actúa como Ingeniero de Datos, Científico de Datos (ML) y experto en crochet. Entrega tres bloques secuenciales: (1) dataset CSV, (2) script de entrenamiento en Python con scikit-learn, (3) exportación a ONNX e inferencia en JavaScript.

### OBJETIVO
Predecir cuántos ovillos (`ovillos_necesarios`) se necesitan para tejer un plaid de ganchillo, comparando KNN, Regresión Lineal y Random Forest, y dejar el mejor modelo funcionando en el navegador.

### REGLAS GENERALES
- Responde en español. Código y comentarios claros, sin relleno.
- Si no estás seguro de un dato de una ficha técnica, NO lo inventes: exclúyelo o márcalo con `confianza_dato = "media"` o `"baja"`.
- Entrega cada bloque por separado, con su nombre de archivo en el encabezado. No continúes al siguiente bloque sin terminar el anterior.

---

## BLOQUE 1: DATASET (`dataset_plaids_crochet.csv`)

### 1.1 Productos reales
- Usa solo marcas y líneas reales (Katia, DROPS, Lion Brand, Paintbox Yarns, Scheepjes, Stylecraft, Patons, DMC, Bernat).
- `peso_ovillo_g`, `metros_ovillo`, `composicion` y `ganchillo_mm` deben coincidir con la ficha técnica oficial. Ejemplos: DROPS Paris = 50 g / 75 m / 5.0 mm; Stylecraft Special DK = 100 g / 295 m / 4.0 mm.
- Usa al menos 12 líneas distintas, de al menos 6 marcas, cubriendo grosores desde DK hasta chunky. Ninguna marca debe superar el 25 % de los registros.

### 1.2 Fórmula obligatoria
```
area_muestra_cm2 = muestra_largo_cm * muestra_ancho_cm
area_plaid_cm2   = plaid_largo_cm * plaid_ancho_cm
gramos_totales   = round(area_plaid_cm2 / area_muestra_cm2 * muestra_gramos, 1)
ovillos_necesarios = ceil(gramos_totales / peso_ovillo_g)
```

### 1.3 Coherencia física
- Muestras estándar de 10x10 cm o 20x20 cm.
- `muestra_gramos` debe ser plausible según el grosor del hilo, la aguja y el punto. Orientación para 10x10 cm: DK ≈ 3-6 g, aran ≈ 5-8 g, chunky ≈ 8-14 g (escala x4 para 20x20).
- Punto bajo consume ≈ 15-20 % más que punto alto en la misma superficie.
- Tamaños de plaid realistas: de bebé (≈ 70x90 cm) a cama doble (≈ 180x220 cm). Mezcla `tipo_punto` entre: punto bajo, punto alto, punto alto doble, granny square, punto de arroz.

### 1.4 Esquema (50 registros, columnas exactas y en este orden)
`id_proyecto, marca, linea_producto, composicion, peso_ovillo_g, metros_ovillo, ganchillo_mm, tipo_punto, muestra_largo_cm, muestra_ancho_cm, muestra_gramos, plaid_largo_cm, plaid_ancho_cm, gramos_totales, ovillos_necesarios, confianza_dato`

### 1.5 Entrega
- Genera en 2 lotes de 25 filas para evitar truncamientos. Entrega el CSV completo y sin comentarios, dentro de un bloque de código.
- Incluye un script `validar_dataset.py` que verifique: 50 filas, columnas exactas, sin nulos, `gramos_totales` y `ovillos_necesarios` recalculados con la fórmula, y valores físicos dentro de rango.
- Lista aparte las filas con `confianza_dato` distinta de "alta" y por qué.

---

## BLOQUE 2: ENTRENAMIENTO (`entrenar_modelos.py`)

### 2.1 Variables
- Objetivo principal: `gramos_totales`; después `ovillos_necesarios_pred = ceil(gramos_pred / peso_ovillo_g)`. Evalúa también la variante que predice directamente `ovillos_necesarios`.
-