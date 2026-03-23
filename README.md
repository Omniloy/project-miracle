# Transcripciones y Audios Clínicos Sintéticos con CIE-10 y SNOMED CT

Dataset sintético de **100 transcripciones clínicas** médico-paciente en español, codificadas con CIE-10 (Internacional), SNOMED CT y ATC, acompañadas de audio generado por IA.

## Descripción

Cada transcripción simula una consulta médica realista estructurada en 5 fases: motivo de consulta, anamnesis, exploración, impresión diagnóstica y plan/despedida. Las transcripciones cubren 10 especialidades médicas comunes en España, con 10 casos por especialidad (8 enfermedades comunes + 2 enfermedades raras).

## Estructura del repositorio

```
Transcripciones_Clinicas_Sinteticas/
├── README.md
├── .env                          # Credenciales ElevenLabs (no incluir en repositorios públicos)
├── metadata/
│   ├── master_cases.json         # Definición maestra de los 100 casos
│   ├── dataset_summary.json      # Resumen estadístico del dataset
│   ├── especialidades.json       # Detalle por especialidad
│   └── codigos_referencia.json   # Referencia de todos los códigos usados
├── transcripciones/
│   ├── cardiologia/              # CAR-001 a CAR-010
│   ├── dermatologia/             # DER-001 a DER-010
│   ├── endocrinologia/           # END-001 a END-010
│   ├── ginecologia/              # GIN-001 a GIN-010
│   ├── medicina_familia/         # MF-001 a MF-010
│   ├── neumologia/               # NEU-001 a NEU-010
│   ├── neurologia/               # NRL-001 a NRL-010
│   ├── pediatria/                # PED-001 a PED-010
│   ├── traumatologia/            # TRA-001 a TRA-010
│   └── urologia/                 # URO-001 a URO-010
├── audios/
│   └── [misma estructura que transcripciones, archivos .mp3]
└── scripts/
    └── generate_audio.py         # Script de generación de audio
```

## Especialidades

| Especialidad | Código | Casos comunes | Enfermedades raras |
|---|---|---|---|
| Medicina de familia | MF | 8 | 2 |
| Pediatría | PED | 8 | 2 |
| Cardiología | CAR | 8 | 2 |
| Dermatología | DER | 8 | 2 |
| Traumatología | TRA | 8 | 2 |
| Ginecología | GIN | 8 | 2 |
| Neumología | NEU | 8 | 2 |
| Neurología | NRL | 8 | 2 |
| Endocrinología | END | 8 | 2 |
| Urología | URO | 8 | 2 |

## Estructura de cada transcripción (JSON)

Cada archivo JSON contiene:

- **id**: Identificador único (ej. MF-001)
- **especialidad**: Nombre de la especialidad
- **resumen**: Resumen clínico del caso
- **diagnostico_principal**: Diagnóstico con CIE-10, SNOMED CT, indicador de enfermedad rara
- **diagnosticos_secundarios**: Lista de diagnósticos adicionales codificados
- **sintomas**: Síntomas con CIE-10 y SNOMED CT
- **procedimientos**: Procedimientos con SNOMED CT
- **medicamentos**: Fármacos con ATC, dosis, posología, SNOMED CT e indicación
- **constantes_vitales**: Solo cuando son clínicamente relevantes
- **notas_adicionales**: Contexto clínico adicional
- **paciente/medico**: Datos ficticios del paciente y médico
- **voces**: IDs de voz ElevenLabs asignados
- **transcripcion**: Diálogo estructurado en 5 fases con audio tags
- **validacion**: Estado de verificación de todos los códigos

## Sistemas de codificación

- **CIE-10 Internacional** (OMS): Todos los códigos son terminales/finales (billable). Se usa la versión internacional, no CIE-10-CM.
- **SNOMED CT**: Concept IDs verificados en el browser oficial de IHTSDO.
- **ATC** (OMS): Códigos del sistema Anatómico Terapéutico Químico verificados en el índice ATC/DDD.

## Audio

Los audios se generaron con la API **ElevenLabs Text to Dialogue** usando el modelo `eleven_v3` desde el endpoint EU residency. Cada archivo MP3 contiene el diálogo completo médico-paciente con entonación contextual basada en audio tags embebidos en el texto.

## Uso previsto

- Entrenamiento y evaluación de modelos de NLP clínico en español
- Extracción automática de diagnósticos CIE-10
- Normalización de terminología clínica con SNOMED CT
- Comprensión conversacional en contextos clínicos
- Reconocimiento de voz médica (ASR)
- Investigación en enfermedades raras

## Licencia y aviso

Este es un dataset **completamente sintético**. Todos los nombres de pacientes y médicos son ficticios. No contiene datos de pacientes reales. Los casos clínicos están diseñados para ser médicamente plausibles pero no deben usarse para diagnóstico clínico real.
