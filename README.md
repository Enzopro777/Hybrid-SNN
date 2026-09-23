# Hybrid-SNN — Sistema de Inteligencia Artificial Neuronal de Spikes

**Versión actual:** 1.6.29f — Readout Stability & Bridge Credit Control  
**Autor:** Enzo ([@Enzopro777](https://github.com/Enzopro777))  
**Licencia:** GPL-3.0  
**Contacto:** enzopro27@gmail.com

---

## ¿Qué es este proyecto?

Un sistema de inteligencia artificial construido desde cero, sin frameworks de deep learning.

La idea central es combinar dos paradigmas que normalmente no se mezclan:

- Un **sustrato de neuronas simuladas que se comunican por pulsos eléctricos** (Spiking Neural Network / SNN), donde el tiempo importa y cada spike es un evento discreto
- Una **organización en bloques funcionales con roles especializados**, inspirada en la arquitectura modular de los modelos de lenguaje (LLM): percepción, representación cortical, workspace latente, readout

El resultado no es una red neuronal clásica ni un transformer. Es un sistema donde la temporalidad biológica del spike convive con la organización modular de la IA moderna.

Desarrollado de forma independiente durante más de un año y medio, corriendo en hardware modesto (CPU, sin GPU).

---

## Arquitectura

```
Entrada visual (pantalla)
        ↓
  [Transductor]          → digitaliza la señal visual en spikes
        ↓
  [Bordes V / Bordes H]  → detección de bordes verticales y horizontales
        ↓
  [Esquinas / Diagonales / Curvas / Simetría / Junctions]
        ↓
  [Detector de Letras]   → integración de features en scores por letra
        ↓
  [Workspace Latente]    → representación recurrente con memoria de estado
        ↓
  [Readout / Language IO]→ decisión física por pools de neuronas
        ↓
     Salida
```

Cada bloque es un módulo Python independiente que se comunica con los demás mediante sinapsis reales — no arrays pasados por función.

---

## Sistemas biológicos implementados

| Sistema | Descripción |
|---|---|
| **Plasticidad STDP** | Las sinapsis aprenden por correlación temporal entre spikes |
| **Modulación dopaminérgica** | El refuerzo positivo amplifica el aprendizaje sináptico |
| **Homeostasis** | Los umbrales de activación se adaptan para evitar hiperactividad |
| **Metabolismo neuronal** | Las neuronas tienen energía; sin recursos, no disparan |
| **Calcio intracelular** | Regula la plasticidad a largo plazo |
| **Crecimiento sináptico** | La red puede generar nuevas conexiones durante el entrenamiento |
| **Competición lateral** | Los pools compiten entre sí; solo uno "gana" por trial |
| **Replay NREM** | El sistema consolida lo aprendido fuera de los trials activos |
| **Currícula adaptativa** | El entrenamiento desbloquea letras nuevas por dominio progresivo |

---

## Estructura del proyecto

```
IA 1.6.29f/
│
├── menu_arranque.py        ← punto de entrada principal
├── main.py                 ← arranque directo (sin menú)
├── config.py               ← todos los parámetros del sistema
├── setup_requirements.py   ← instala dependencias automáticamente
│
├── core/                   ← red neuronal base
│   ├── network.py          ← NeuralNetwork: neuronas, sinapsis, spikes
│   ├── region.py           ← regiones corticales y organización espacial
│   ├── reward_system.py    ← sistema de recompensa por dopamina
│   └── ...
│
├── engine/                 ← motor de simulación
│   ├── simulation.py       ← loop principal, trials, telemetría
│   └── thread_manager.py   ← ejecución multi-hilo
│
├── modules/                ← bloques funcionales especializados
│   ├── vision.py           ← captura y procesamiento visual
│   ├── transductor_block.py
│   ├── detector_bordes_v/h.py
│   ├── detector_letras.py
│   ├── language_io.py      ← entrada/salida de símbolos
│   ├── causal_probe.py     ← diagnóstico causal de la red
│   ├── curriculum.py       ← currícula adaptativa
│   └── ...
│
├── systems/                ← subsistemas biológicos
│   ├── plasticity.py       ← STDP y aprendizaje sináptico
│   ├── homeostasis.py      ← estabilidad de activación
│   ├── metabolism.py       ← energía neuronal
│   ├── latent_workspace.py ← workspace latente recurrente
│   ├── growth.py           ← crecimiento sináptico
│   └── ...
│
└── tests/                  ← suite de tests (35+ archivos)
```

---

## Instalación y uso

**Requisitos:** Python 3.9+

```bash
# 1. Clonar el repositorio
git clone https://github.com/Enzopro777/IA-SNN.git
cd "IA 1.6.29f"

# 2. Instalar dependencias
python setup_requirements.py

# 3. Arrancar el sistema
python menu_arranque.py
```

El menú de arranque ofrece:
- **[A]** Iniciar normalmente (carga calibración guardada si existe)
- **[B]** Recalibrar y arrancar (warmup automático + ajuste de umbrales)
- **[C]** Ajuste manual de umbrales
- **[D]** Solo calibrar (diagnóstico sin correr la red)
- **[E]** Ver estado de detectores

> **Nota:** La versión 1.6.29f está configurada con `CLEAN_EXPERIMENT_DEFAULT = True`. Arranca sin estado previo por defecto. Para reanudar aprendizaje existente, cambiar ese parámetro en `config.py`.

---

## Estado actual (v1.6.29f)

El sistema reconoce actualmente cinco letras: **X, O, T, A, E**.

- Aprendizaje sináptico funcional (STDP + dopamina operativos)
- Representación interna distinguible entre clases en el workspace latente
- Trabajo activo en la selectividad del readout físico (bridge H0/Latent → pools)
- CausalProbe operativo: permite aislar qué neuronas contribuyen a cada decisión

Esto es un sistema experimental en desarrollo activo, no un producto terminado.

---

## Cómo contribuir

Toda contribución es bienvenida. El proyecto está documentado en español.

Áreas donde hay trabajo por hacer:
- Mejorar la selectividad del readout (cuello de botella actual)
- Ampliar el vocabulario más allá de 5 letras
- Optimizar el rendimiento en CPU (actualmente ~1-2 trials/minuto en i5)
- Documentar los módulos individuales

Para proponer cambios: abrí un issue describiendo qué querés modificar y por qué, antes de hacer un PR.

---

## Contexto del proyecto

Este sistema no nació de un paper ni de un curso. Nació de la pregunta:

*¿Se puede construir una IA que aprenda de forma más parecida a cómo funciona un sistema neuronal real?*

La combinación de biología (plasticidad, homeostasis, metabolismo), arquitectura modular tipo LLM y experimentación iterativa produjo algo que no encaja del todo en ninguna categoría existente. Eso es, para bien o para mal, exactamente lo que se buscaba.

