@echo off
echo ================================================================================
echo GNN PROPER CLASSIFICATION - NO TARGET LEAKAGE
echo ================================================================================
echo.
echo This will train GNN WITHOUT target leakage features
echo Features excluded: Days for shipping (real), Delivery Status
echo.
echo Expected results: 60-75%% accuracy (realistic, not perfect)
echo Expected time: 2-3 minutes
echo ================================================================================
echo.

call venv_ppo\Scripts\activate.bat
python fix_gnn_proper_classification.py

echo.
echo ================================================================================
echo Check results in: results\delay_classification_gnn\
echo ================================================================================
pause















