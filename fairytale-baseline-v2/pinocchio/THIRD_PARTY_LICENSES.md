# Componenti di terze parti

- **Qwen/Qwen2.5-0.5B-Instruct**: Alibaba Cloud, Apache License 2.0. La licenza integrale accompagna il checkpoint in `models/corrected/LICENSE`. L'originale completo rimane nel repository separato `../../little-red-weightspace/models/original`, alla revisione `7ae557604adf67be50417f59c2c2f167def9a775`. Il checkpoint corretto è una modifica sperimentale, non un rilascio ufficiale Qwen; `patches/current.json` ne identifica la versione. Il nome `corrected` non certifica un risultato positivo.
- **llama.cpp**: MIT, https://github.com/ggml-org/llama.cpp. Convertitore esterno, commit `5851a8e505f4d5fbb183f2847698eb6155d176b7`; non incluso qui.
- **PyTorch**: BSD-style; **Transformers, Hugging Face Hub e Safetensors**: Apache 2.0; **NumPy**: BSD; **Requests**: Apache 2.0. Le dipendenze non sono redistribuite; versioni fissate in `requirements.txt`.
- **Ollama**: MIT, programma esterno usato per il confronto dei checkpoint.
- Riferimento canonico: Carlo Collodi, *Le avventure di Pinocchio*, Bemporad 1902, trascrizione Project Gutenberg 52484. Il testo integrale non è incluso. Le domande sono state scritte per questo esperimento.
- Il lavoro MIT su Weight-Space Programming è citato come fonte scientifica e non viene redistribuito.

La licenza MIT nella radice riguarda il codice e i materiali originali dell'esperimento. Non sostituisce le licenze dei pesi e delle dipendenze.
