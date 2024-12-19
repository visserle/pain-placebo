﻿# Check for PowerShell version 6.0 or higher
if ($PSVersionTable.PSVersion.Major -lt 6) {
    Write-Host "Please install PowerShell 6.0 or higher to run this script."
    Read-Host
    exit
}

# Initialize Conda environment
if ($IsWindows) {
    # Windows specific Conda initialization
    $condaPath = "C:\mamba"  # Change this to the path of your Conda installation
    & "$condaPath\shell\condabin\conda-hook.ps1"
    # Depending on the Conda setup you might need this instead to initialize Conda
    # & "$condaPath\Scripts\activate.ps1"
    conda activate pain
}
elseif ($IsMacOS) {
    # macOS specific Conda initialization (only used for development without iMotions and the respective dummy_imotions flag)
    $condaPath = "$HOME/miniforge3"
    # Use bash to activate the environment and get the Python path
    $pythonPath = /bin/bash -c "source '$condaPath/bin/activate' pain; which python"
    
    # Extract the directory path of the Python executable
    $pythonDir = Split-Path -Parent $pythonPath
    
    # Prepend the Python directory to the PATH environment variable
    $env:PATH = "$pythonDir" + ":" + $env:PATH
}

# Print the welcome message
Write-Host "*** DEVELOP MODE ***"
Write-Host "" 
Write-Host "This is the behavioral pain treatment experiment."
Write-Host ""
Write-Host ""
Write-Host "The experiment consists of four parts:"
Write-Host "1. Pre-Experiment Questionnaires"
Write-Host "2. Pain Calibration"
Write-Host "3. Pain Treatment"  # = Pain Placebo
Write-Host "4. Post-Experiment Questionnaires"
Write-Host ""
Write-Host ""
Write-Host "Press [Enter] to start..."
Read-Host

# Execute Python scripts
Write-Host "1. Pre-Experiment Questionnaires"
python -m src.experiments.add_participant
#python -m src.experiments.questionnaires.app general bdi-ii phq-15 panas --welcome #--debug
Write-Host ""
Write-Host "Press [Enter] to continue with the calibration..."
Read-Host

Write-Host "2. Pain Calibration"
#python -m src.experiments.calibration.calibration --windowed --debug #--dummy_thermoino --dummy_stimulus # --windowed --debug

Write-Host ""
Write-Host "Press [Enter] to continue with the treatment..."
Read-Host

Write-Host "3. Pain Treatment"
python -m src.experiments.placebo.placebo
Write-Host ""
Write-Host "Press [Enter] to continue with the questionnaires..."
Read-Host

Write-Host "4. Post-Experiment Questionnaires"
python -m src.experiments.questionnaires.app panas pcs pvaq stai-t-10 maas --debug
# Print the completion message
Write-Host "Experiment completed."
Write-Host ""
Write-Host "╔═════════════════════════════════════════════════════════════════════════════╗"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "║    ██████╗  ██████╗  ██████╗ ██████╗          ██╗ ██████╗ ██████╗     ██╗   ║"
Write-Host "║   ██╔════╝ ██╔═══██╗██╔═══██╗██╔══██╗         ██║██╔═══██╗██╔══██╗    ██║   ║"
Write-Host "║   ██║  ███╗██║   ██║██║   ██║██║  ██║         ██║██║   ██║██████╔╝    ██║   ║"
Write-Host "║   ██║   ██║██║   ██║██║   ██║██║  ██║    ██   ██║██║   ██║██╔══██╗    ╚═╝   ║"
Write-Host "║   ╚██████╔╝╚██████╔╝╚██████╔╝██████╔╝    ╚█████╔╝╚██████╔╝██████╔╝    ██╗   ║"
Write-Host "║    ╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝      ╚════╝  ╚═════╝ ╚═════╝     ╚═╝   ║"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
Read-Host
exit
