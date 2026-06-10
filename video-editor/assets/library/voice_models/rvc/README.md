# Modelos RVC locales

Coloca aqui solo modelos con permiso/legalmente utilizables.

Estructura recomendada:

```text
rvc/
  mi_voz/
    model.pth
    added.index
    voice.json
```

`voice.json` es opcional:

```json
{
  "name": "Mi voz RVC",
  "pitch": 0,
  "f0_method": "rmvpe"
}
```

Para activar inferencia, configura `WORKFAST_RVC_COMMAND` en `.env` con el
comando de tu instalacion Applio/RVC. WorkFast reemplaza estos placeholders:
`{input}`, `{output}`, `{model}`, `{index}`, `{pitch}`, `{f0_method}`,
`{device}`, `{preset}` y `{voice_id}`.
