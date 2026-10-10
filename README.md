# Agent Mistey

**Team:** Agentic Lab  
**Theme:** MedTech / BioTech / HealthTech  
**Category:** Software

## Overview

Agent Mistey is an AI-powered assistive system designed to enable hands-free computer access through head and face tracking, hand gestures, and voice commands. It aims to make everyday computing more accessible while exploring how natural user interactions can provide useful motor-function insights over time.

Mistey is designed with an offline-first approach, privacy in mind, and compatibility with ordinary computing devices.

## Key Features

* **Hands-free computer control:** Interact with a computer using head/face movement, hand gestures, and voice.
* **Multimodal interaction:** Multiple input pathways support different accessibility needs.
* **Motor-function signal analysis:** Explore movement characteristics such as tremor, smoothness, target-acquisition behaviour, and changes across sessions.
* **Offline-first design:** Core interaction is designed to work locally without relying on cloud inference.
* **Low-cost hardware approach:** Designed for ordinary laptops with a webcam and microphone, rather than requiring dedicated eye-tracking hardware.
* **Personalised assistance:** Future development may improve adaptation to individual movement patterns and preferences.

## How It Works

1. **Capture movement and commands** through supported input methods.
2. **Process inputs** using computer-vision and speech-recognition components.
3. **Understand intent and assist** by translating user input into computer actions.
4. **Analyse movement signals** for research-oriented motor-function insights.
5. **Explore longitudinal trends** across sessions, subject to consent and appropriate validation.

## Technology Stack

The project documentation describes the following technologies and components:

* **Language:** Python
* **Computer vision:** OpenCV, MediaPipe
* **Speech recognition:** Whisper / faster-whisper and Vosk fallback options
* **Signal processing:** NumPy
* **Desktop interface:** PyQt5
* **Optional local Q\&A:** Ollama
* **Reporting:** PyQt5 document and printing components
* **Testing:** pytest

Exact components available may vary by build and configuration.

## Potential Impact

* Support greater digital independence for people with motor disabilities.
* Reduce reliance on specialised assistive hardware.
* Make hands-free computing more accessible in homes and educational settings.
* Explore continuous, real-world movement insights that may complement periodic assessment.
* Support future research into accessible, privacy-conscious assistive technology.

## Future Scope

* Clinical and real-user validation with rehabilitation professionals.
* Improved tremor, movement-smoothness, and fatigue analysis.
* Personalised interaction and adaptive accessibility.
* Support for more Indian languages.
* Optimisation for low-cost, low-power devices.
* Longitudinal reporting and rehabilitation-centre integration.
* Scalable deployment across homes, schools, colleges, and rehabilitation centres.

## Research and Validation Note

**Agent Mistey's motor-function measurement capability is a research direction and is not a clinically validated diagnostic instrument.** The project should not be presented as diagnosing Parkinson's disease or any other condition, or as replacing clinical assessment. Real-user and clinician evaluation is a necessary next step.

## Privacy and Design Principles

* Prioritise local processing for core interaction and motor-signal handling.
* Minimise unnecessary collection or transmission of personal data.
* Make consent and privacy expectations clear for any research data collection.
* Distinguish observed movement signals from clinical conclusions.


## Team

|Member|Domain / Contribution|
|-|-|
|Partho Protim Roy|Team Lead; Computer Science, system architecture, AI and integration|
|Mathias Oundo|Rehabilitation and human-movement perspective|
|Anindya Bagchi|Computer vision and software engineering|
|Soumik Biswas|Frontend and accessibility user experience|
|Tapu Mondal|Testing, integration and reliability|
|Bishakha Poddar|Biomedical research and scientific literature|

## 

## How to Install?
You can Use the software by downloading from Website...

## Project Links (recomended)

* **Website:** https://mistey.provatsoft.com
* Tutorial Video. 
  * Doctor Review: https://drive.google.com/file/d/1G6_WXEKoECqOBQ3VoXT3Bcq3Fk5C7qT4/view?usp=drive_link
  * How to Use: https://drive.google.com/file/d/1htYI_kNaGmSxFsDqS3xobdDK6M2drYET/view?usp=drive_link
  
* **Team:** Agentic Lab


## Acknowledgements

This project draws on existing research into digital interaction and motor-function signals, accessibility technologies, and relevant disability-inclusion and education policies.

\---

*Agent Mistey — Assistive interaction today, evidence-based accessibility tomorrow.*

