# TODO operativo finale

1. Congelare con Git lo stato attuale funzionante.

2. Pulire e finalizzare `optimization_constraint_aware.py`.

3. Salvare bene risultati e figure della comparison:
   - baseline vs constraint-aware
   - target error
   - constraint violation
   - max_xr
   - feasibility rate

4. Scegliere la soluzione finale constraint-aware da usare nel resto del progetto:
   - preferibilmente target `0.64`
   - alternativa conservativa: target `0.62`

5. Creare/salvare un file con la soluzione finale selezionata.

6. Aggiornare `visualization.py` per usare la soluzione finale constraint-aware.

7. Rigenerare:
   - trajectory finale
   - GIF/animation finale
   - plot `y(t)`, `x_r(t)`, `x_b(t)`

8. Rifare sensitivity analysis sulla soluzione finale constraint-aware.

9. Rifare Monte Carlo robustness sulla soluzione finale constraint-aware.

10. Controllare se la soluzione finale è robusta:
    - feasibility rate
    - max constraint violation
    - mean target error
    - 5th–95th percentile outreach

11. Se target `0.64` non è abbastanza robusto, passare al target `0.62`.

12. Fare learning curve analysis.

13. Fare confronto GP vs Random Forest.

14. Sistemare e organizzare risultati cross-validation già ottenuti.

15. Fare confronto optimizer:
    - Random Search
    - Powell Multi-Start
    - Differential Evolution

16. Decidere il setup finale:
    - surrogate finale
    - objective finale
    - optimizer finale
    - target finale per analisi dettagliata

17. Pulire cartelle `results/` e `figures/`.

18. Aggiornare README con:
    - pipeline finale
    - struttura report
    - TODO/future work

19. Iniziare scrittura report:
    - Problem Statement
    - Dynamic Simulator
    - Dataset Generation
    - Targeted Augmentation

20. Completare report con:
    - Learning Curve
    - Surrogate Modeling
    - Constraint-Aware Optimization
    - Physical Validation
    - Sensitivity
    - Robustness
    - Conclusions

21. Preparare presentazione/slides.

22. Solo se avanza tempo: valutare Bayesian Optimization come extra/future work.