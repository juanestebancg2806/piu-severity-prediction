# Predicción de la severidad del PIU

Parte del curso "Proyecto I de Innovación Tecnológica en Inteligencia Artificial", Maestría en Inteligencia Artificial Aplicada, Universidad Icesi, Cali, Colombia.

## Estado del proyecto

Activo

## Integrantes

**Profesor:** [Milton Orlando Sarria](https://github.com/miltonsarria)

| Nombre |
| --- |
| Katherin Adriana Camargo Cetina |
| Juan Esteban Cardona García |
| Juan David Martínez Legarda |
| Daniel Velasco López |

## Introducción y objetivo del proyecto

El uso problemático de internet (PIU, por sus siglas en inglés) en niños y adolescentes se asocia con depresión, ansiedad y alteraciones del sueño; sin embargo, suele detectarse tarde: su evaluación depende de valoraciones clínicas especializadas que son costosas y poco accesibles para muchas familias. En cambio, las mediciones de actividad física y condición física son fáciles de obtener y se recogen ampliamente.

El objetivo de este proyecto es evaluar qué tan bien los indicadores de actividad física y condición física, junto con variables demográficas básicas, permiten estimar la severidad del PIU, medida con el Severity Impairment Index (SII), en niños y adolescentes de la Healthy Brain Network, mediante aprendizaje automático supervisado. Se busca determinar si estos indicadores podrían servir de base para una herramienta de tamizaje temprana y accesible. El proyecto evalúa la viabilidad; no construye una herramienta diagnóstica.

## Métodos utilizados

- Modelo de proceso CRISP-DM
- Análisis exploratorio de datos
- Pipelines de preprocesamiento sin fuga de información: indicadores de datos faltantes por instrumento, imputación y transformación Yeo-Johnson ajustadas dentro de cada pliegue de entrenamiento
- Clasificación ordinal: línea base multiclase, descomposición de Frank y Hall, y regresión con umbrales optimizados
- Regresión logística regularizada, bosques aleatorios, gradient boosting y máquinas de vectores de soporte
- Validación cruzada anidada, repetida y estratificada
- Kappa ponderado cuadrático (QWK), sensibilidad por nivel y tasa de subestimación
- Pruebas de permutación de etiquetas y comparaciones pareadas por pliegue con la prueba t corregida de Nadeau–Bengio
- Interpretabilidad del modelo con valores SHAP
- Análisis de errores por sexo y grupo de edad

## Tecnologías

- Python
- uv
- pandas, NumPy, PyArrow
- scikit-learn
- SHAP
- Matplotlib, seaborn

## Descripción del proyecto

**Datos.** El proyecto usa el conjunto de datos [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) de Kaggle, derivado de la Healthy Brain Network. El conjunto de entrenamiento contiene 3.960 participantes de 5 a 22 años; 2.736 de ellos tienen etiqueta SII. Los datos no se redistribuyen en este repositorio; consulta [data/README.md](data/README.md) para obtenerlos.

**Variable objetivo.** El SII es una variable ordinal con cuatro niveles (ninguno, leve, moderado, severo), derivada del Parent-Child Internet Addiction Test. Las clases están muy desbalanceadas: solo 34 participantes etiquetados se ubican en el nivel severo.

**Conjuntos de variables.** Se comparan tres conjuntos anidados de variables con los mismos participantes y las mismas particiones de validación cruzada:

| Conjunto | Variables |
| --- | --- |
| Referencia | Demográficas (edad, sexo, temporada de inscripción) |
| Principal | Demográficas + medidas físicas (antropometría, signos vitales, FitnessGram, impedancia bioeléctrica, cuestionario de actividad física, actigrafía de muñeca) |
| Complementario | Principal + alteraciones del sueño, funcionamiento global y horas de uso de internet |

Una ablación del conjunto principal sin las variables de actigrafía aísla el aporte de los datos del dispositivo portátil.

**Enfoque.** La búsqueda de modelos se hace sobre el conjunto principal. Luego, la configuración seleccionada se entrena con los conjuntos de referencia y complementario, de modo que las diferencias de desempeño reflejen la información que aportan las variables y no el algoritmo. La estrategia de datos faltantes y los hiperparámetros se seleccionan dentro del ciclo interno de la validación cruzada para evitar fuga de información, y se excluyen las columnas PCIAT que definen la variable objetivo.

**Evaluación.** Como el QWK es simétrico, se complementa con la sensibilidad por nivel y con la tasa de subestimación de los casos moderados y severos, que es el error más costoso en un contexto de tamizaje.

**Alcance y ética.** Los datos provienen de menores de edad en una muestra clínica, no representativa, del área de Nueva York, y la variable objetivo se basa en el reporte de los padres. Los resultados tienen fines exclusivamente académicos y no constituyen una herramienta diagnóstica.

## Primeros pasos

1. Clona el repositorio.
2. Instala [uv](https://docs.astral.sh/uv/).
3. Ejecuta `uv sync`. Esto instala las dependencias y el paquete `piu_severity` en modo editable.
4. Obtén los datos como se describe en [data/README.md](data/README.md).

Quien no use uv puede generar un `requirements.txt` con:

```bash
uv export --format requirements-txt > requirements.txt
```

## Cómo contribuir

Todos los integrantes del equipo son colaboradores: clonen el repositorio, trabajen en una rama `feature/*` y abran un pull request hacia `main`. El flujo de trabajo paso a paso está en [CONTRIBUTING.md](CONTRIBUTING.md). Las reglas para colaboradores y asistentes de programación con IA (privacidad de datos, prevención de fuga de información, arquitectura y estilo de código) están definidas en [AGENTS.md](AGENTS.md); `CLAUDE.md` importa ese mismo archivo.

## Estructura del proyecto

```
piu-severity-prediction/
├── README.md                 # Descripción general del proyecto
├── AGENTS.md                 # Reglas para colaboradores y asistentes de IA
├── CLAUDE.md                 # Importa AGENTS.md para Claude Code
├── CONTRIBUTING.md           # Flujo de trabajo en Git del equipo
├── .gitignore
├── .python-version           # Versión de Python fijada para uv
├── pyproject.toml            # Metadatos y dependencias del proyecto
├── uv.lock                   # Versiones exactas de las dependencias (reproducibilidad)
├── LICENSE                   # Licencia MIT (solo el código; los datos tienen su propia licencia de Kaggle)
├── deliverables/             # Documentos entregados en cada hito del curso
│   ├── README.md
│   ├── deliverable-1/        # Documentos del hito 1
│   ├── deliverable-2/        # Documentos del hito 2
│   └── deliverable-3/        # Documentos del hito 3
├── docs/                     # Documentación técnica transversal
│   └── README.md
├── references/               # Bibliografía y material de consulta
├── data/                     # Datos locales (no versionados)
│   ├── README.md
│   ├── raw/                  # Datos originales inmutables
│   ├── interim/              # Transformaciones intermedias
│   └── processed/            # Datos finales listos para modelar
├── notebooks/                # Exploración y narrativa
│   └── README.md
├── src/
│   └── piu_severity/         # Paquete instalable
│       ├── __init__.py
│       ├── config.py         # Rutas y configuración centralizadas
│       ├── data/             # Carga, descarga y limpieza
│       ├── features/         # Grupos de variables, ingeniería de variables y preprocesamiento
│       ├── models/           # Pipelines de modelos y estimadores ordinales
│       ├── evaluation/       # Métricas, validación cruzada y comparación de modelos
│       └── visualization/    # Funciones de graficación reutilizables
├── models/                   # Modelos entrenados
├── reports/
│   └── figures/              # Figuras generadas
└── tests/                    # Pruebas automatizadas
```

## Entregables destacados

- [Entregable 1](deliverables/deliverable-1/) — Formulación del proyecto: análisis del problema, estado del arte, árbol de problemas, objetivos, metodología propuesta y plan del semestre (19 de octubre de 2026).
- [Entregable 2](deliverables/deliverable-2/) — Comprensión de los datos y experimentos iniciales: análisis exploratorio, tratamiento de los datos, pipeline de validación, modelo de línea base y pruebas de hipótesis iniciales (14 de noviembre de 2026, planeado).
- [Entregable 3](deliverables/deliverable-3/) — Informe final, video y presentación oral: modelado, evaluación, interpretabilidad y recomendaciones de viabilidad (1 de diciembre de 2026, planeado).
