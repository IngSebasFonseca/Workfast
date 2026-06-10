' Evse Video Studio - launcher (sin CMD, sin nada visible).
'
' Lanza pythonw.exe (Python sin consola) ejecutando launcher_window.py.
' launcher_window.py se encarga de:
'   - spawnar el server Flask como subproceso oculto
'   - abrir la ventana nativa WebView2
'   - cerrar el server cuando se cierra la ventana
'
' Todo el setup pesado (venv + pip install) lo hizo el instalador,
' asi que aca no hay nada que verificar. Click -> ventana nativa. Listo.

Option Explicit

Dim fso, wsh, scriptDir, pythonw, launcher

Set fso = CreateObject("Scripting.FileSystemObject")
Set wsh = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
wsh.CurrentDirectory = scriptDir

' venv: preferir el local (copia de desarrollo) si existe; sino el estable
' por-usuario que crea el instalador (sobrevive reinstalaciones).
Dim localVenv, stableVenv
localVenv = scriptDir & "\venv\Scripts\pythonw.exe"
stableVenv = wsh.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\EvseVideoStudio\venv\Scripts\pythonw.exe"
If fso.FileExists(localVenv) Then
    pythonw = localVenv
Else
    pythonw = stableVenv
End If
launcher = scriptDir & "\launcher_window.py"

If Not fso.FileExists(pythonw) Then
    MsgBox _
        "No encuentro venv\Scripts\pythonw.exe en:" & vbCrLf & scriptDir & vbCrLf & vbCrLf & _
        "El programa no esta bien instalado. Desinstala y vuelve a correr el setup.", _
        vbCritical, "Evse Video Studio"
    WScript.Quit 1
End If

If Not fso.FileExists(launcher) Then
    MsgBox _
        "No encuentro launcher_window.py en:" & vbCrLf & scriptDir & vbCrLf & vbCrLf & _
        "Archivos del programa incompletos. Reinstala.", _
        vbCritical, "Evse Video Studio"
    WScript.Quit 1
End If

' pythonw.exe nunca abre consola. Show=0 + False (no esperar).
wsh.Run """" & pythonw & """ -B """ & launcher & """", 0, False
