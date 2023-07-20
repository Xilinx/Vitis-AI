echo off 
REM ##################################################################################################
REM #                                                                                                #
REM #    Script: make_gh_pages.bat                                                                   #
REM #    Purpose: Update release docs from current branch to gh-pages branch                         #
REM #    Usage:                                                                                      #
REM #      - Execute the command 'make_gh_pages' from current branch                                 #
REM #      - If current branch is 'master' you must provide the release number                       #
REM #      - 'gh-pages' dir structure must match the below                                           #
REM #      - Each branch including gh-pages must have a copy of this file in the same directory path #
REM #                                                                                                #
REM # 	       root                                                                                  #
REM #           ├── release1                                                                         #
REM #           │   └─ html                                                                          #
REM #           │      └── ...                                                                       #
REM #           │                                                                                    #
REM #           ├── release2                                                                         #
REM #           │   └─ html                                                                          #
REM #           │      └── ...                                                                       #
REM #          ...                                                                                   #
REM #                                                                                                #
REM #    Revision history                                                                            #
REM #          2023-07-02  First release                                                             #
REM # 	       2023-07-18  Whitespace cleanup                                                        #
REM #                                                                                                #
REM #    Copyright (c) 2023 Advanced Micro Devices, Inc.                                             #
REM #                                                                                                # 
REM ##################################################################################################
                                                                     
for /f %%i in ('git symbolic-ref --short -q HEAD') do set CUR_BRANCH=%%i
echo Entering gh-pages check

if %CUR_BRANCH%==gh-pages (
	echo ####################################################
	echo ERROR: This script cannot be executed from gh-pages.
	echo ####################################################	
	echo. 
	echo Please switch to a branch for which the gh-pages documentation needs to be updated and execute this script from that branch
	echo Exiting scipt, status incomplete	
	exit /b 0 )
	
echo Entering master check
if %CUR_BRANCH%==master (
	echo The {current branch} is %CUR_BRANCH%
	echo Vitis AI GithubIO convention requires that you enter a release number, ie 3.5, 4.0, which will be used as the GithubIO folder name on the gh-pages branch
	echo The contents of {current_branch}/docs will be copied to gh-pages /{release}/html.
	echo ###########################################################################
	echo ERROR:  Master is not a valid release number for Vitis AI GithubIO gh-pages
	echo ###########################################################################
	echo.
	set /P CUR_BRANCH="What is the current release version for this branch? "
	)

echo.
echo You have configured this script to copy documentation to gh-pages /docs/%CUR_BRANCH%/html
set AREYOUSURE=N
set /P AREYOUSURE="Are you sure (Y/[N])?"
if '%AREYOUSURE%' NEQ 'Y' (
	echo Exiting scipt, status incomplete	
	exit /b 0
	)

if %CUR_BRANCH%==master (
	echo ###############################################################################################
	echo ERROR: Master is not a valid release number.  Exiting script to prevent improper gh-pages build
	echo ###############################################################################################
	echo.
	echo Exiting scipt, status incomplete	
	exit /b 0 
	)

echo The current branch is %CUR_BRANCH%
echo Updating gh-pages for %CUR_BRANCH% branch
echo Creating gh_assemble directory
md %TEMP%\gh_assemble
md %TEMP%\gh_assemble_cur
echo Copying current branch HTML files to gh_assemble_cur
robocopy /MIR /njh /njs /ndl /nc /ns ../docs %TEMP%/gh_assemble_cur/%CUR_BRANCH%/html
echo Checkout gh-pages
git checkout gh-pages

for /f %%i in ('git symbolic-ref --short -q HEAD') do set TEST_BRANCH=%%i
if %TEST_BRANCH% NEQ gh-pages (
	echo ###################################################################################
	echo ERROR: gh-pages checkout failed.  Did you have untracked changes on current branch?
	echo ###################################################################################
	echo.
	echo Exiting scipt, status incomplete	
	exit /b 0 
	)

echo Mirror to gh_assemble
robocopy /MIR /njh /njs /ndl /nc /ns ../ %TEMP%/gh_assemble/
robocopy /MIR /njh /njs /ndl /nc /ns %TEMP%/gh_assemble_cur/%CUR_BRANCH%/html %TEMP%/gh_assemble/%CUR_BRANCH%/html
echo Copying Updates to gh-pages
robocopy /MIR /njh /njs /ndl /nc /ns %TEMP%/gh_assemble/ ../
echo Adding files to gh-pages branch
git add --all
echo Commiting changes
git commit -m "Commit by gh-pages auto-build"

set AREYOUSURE=N
set /P AREYOUSURE="Delete temporary files? (Y/[N])?"
if '%AREYOUSURE%'=='Y' (
	echo Deleting temporary files
	rm -rf %TEMP%/gh_assemble/
	rm -rf %TEMP%/gh_assemble_cur/
	)

echo Done! Please push changes to remote origin and submit a pull request
echo You are currently on the gh-pages branch
ls





