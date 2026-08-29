Option Explicit

' Launch one existing scheduler batch file without creating a visible
' console window.  The task waits for completion and receives the batch
' file's exit code, preserving Task Scheduler failure reporting.
If WScript.Arguments.Count <> 1 Then
    WScript.Quit 2
End If

Dim shell, batchPath, command, exitCode
Set shell = CreateObject("WScript.Shell")
batchPath = WScript.Arguments(0)
command = shell.ExpandEnvironmentStrings("%ComSpec%") _
    & " /d /c " & Chr(34) & Chr(34) & batchPath & Chr(34) & Chr(34)

exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode
