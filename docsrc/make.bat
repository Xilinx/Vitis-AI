@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

if "%1" == "github" (
    %SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
    copy %SOURCEDIR%\docs\reference\ModelZoo_VAI3.0_Github_web.htm %BUILDDIR%\html\docs\reference\ModelZoo_VAI3.0_Github_web.htm
	copy %SOURCEDIR%\docs\404.html %BUILDDIR%\html\404.html
	rm ../docs -r
	robocopy %BUILDDIR%/html ../docs /E > nul
	robocopy %SOURCEDIR%/docs/reference/images ../docs/_images /E > nul
    echo.Generated files copied to ../docs
    goto end
)


%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
