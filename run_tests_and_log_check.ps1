# PowerShell script: run_tests_and_log_check.ps1
# Runs all Python tests and checks trading.log for errors/warnings

Write-Host "Running all Python tests with pytest..." -ForegroundColor Cyan
pytest

Write-Host "\nChecking trading.log for ERROR or CRITICAL entries..." -ForegroundColor Cyan
if (Test-Path trading.log) {
    $errors = Select-String -Path trading.log -Pattern 'ERROR','CRITICAL'
    if ($errors) {
        Write-Host "\nFound the following ERROR/CRITICAL log entries:" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host $_.Line -ForegroundColor Red }
    } else {
        Write-Host "\nNo ERROR or CRITICAL entries found in trading.log." -ForegroundColor Green
    }
} else {
    Write-Host "trading.log not found. Run main.py to generate logs." -ForegroundColor Yellow
}

Write-Host "\nAutomation complete." -ForegroundColor Cyan
