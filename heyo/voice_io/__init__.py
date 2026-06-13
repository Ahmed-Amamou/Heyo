"""Server-side voice I/O: STT (faster-whisper) + TTS (Piper).

Lives on the server, next to the models, so voice clients stay thin — a mic,
a speaker, and HTTP. Imported lazily by the /voice endpoints.
"""
