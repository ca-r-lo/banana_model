@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo.
echo Building AgriVision Server Windows Executable...
echo This may take a few minutes as TensorFlow is very large. Please wait...
echo.

pyinstaller --name "AgriVisionServer" --onefile --add-data "templates;templates" --add-data "static;static" --add-data "banana_model.tflite;." app.py

echo.
echo =======================================================
echo BUILD COMPLETE!
echo You can find your AgriVisionServer.exe inside the "dist" folder.
echo =======================================================
pause
