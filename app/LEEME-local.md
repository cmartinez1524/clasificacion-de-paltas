# Paltas Hass: demostración local para Windows

Para computadores Windows 10/11 de 64 bits (x64).

1. Extrae **todo** el ZIP, por ejemplo en `C:\PaltasDemo`. No ejecutes desde dentro del ZIP.
2. Ejecuta `INSTALAR.cmd` una vez en cada computador. Necesita Internet para descargar Python y las dependencias; no requiere instalar Python a mano ni reentrenar.
3. Conecta la cámara USB y ejecuta `INICIAR.cmd`. Se abre `http://localhost:7860` en el navegador de ese computador.
4. En **Cámara en vivo**, permite acceso a la cámara y selecciona la cámara conectada en el control del navegador. Detén la cámara cuando termines.
5. En **Foto**, carga una imagen o prueba uno de los ejemplos incluidos. Verás el estado de madurez, las probabilidades y el mapa de calor descargable.

La carpeta `checkpoints/` incluye los parámetros entrenados. Conserva toda la carpeta de la demo cuando la copies a otro equipo. No necesita el dataset completo y, tras instalarse, puede hacer inferencia sin Internet. Las fotos se procesan en el computador; no usa túneles ni servicios de clasificación externos.

CPU es el modo de instalación predeterminado y no requiere GPU. La velocidad en vivo depende del equipo; si resulta lenta, desmarca el mapa de calor o usa fotos individuales.

Para un computador con NVIDIA compatible y driver actualizado, puedes instalar el entorno CUDA opcional:

```powershell
.\INSTALAR.cmd -Dispositivo NVIDIA
```

Para forzar CPU aunque hayas instalado CUDA:

```powershell
.\INICIAR.cmd -CPU
```

Mantén abierta la ventana de la app. Para cerrarla, pulsa Ctrl+C en esa ventana. Si el puerto 7860 está ocupado, cierra la instancia anterior.

Usa una sola palta Hass centrada, bien iluminada y, preferentemente, sobre fondo blanco. El modelo aprendió con fotos de laboratorio: no mide firmeza ni el estado interno de la fruta. El mapa de calor es una aproximación de las regiones asociadas a la predicción, no una segmentación.

`modelo.json` registra la identidad, el hash SHA-256 y las métricas del modelo. Los cinco ejemplos provienen del test del dataset de Xavier, Rodrigues y Silva (2024), DOI 10.17632/3xd9n945v8.1, CC BY 4.0.
