# Reproduction entry points for "A Unified Geometric Framework for Understanding Collider Event Manifolds".
#
#   make figures          Tier 1: every paper figure and Table 1 from the tracked results/ (seconds)
#   make fig3             one figure (fig1 ... fig8, table1)
#   make sec4             Tier 2: recompute a section's results from data/blocks, then its figures
#
# Set PYTHON to use a specific interpreter, e.g. `make figures PYTHON=python3`.
PYTHON ?= python
export PYTHON

.PHONY: figures fig1 fig2 fig3 fig4 fig5 fig6 fig7 fig8 table1 sec2 sec3 sec4 sec5 sec6 clean-figures

figures: fig1 fig2 fig3 fig4 fig5 fig6 fig7 fig8

fig1:
	$(PYTHON) paper/sec2_metrics/fig1_transport_plans.py
fig2:
	$(PYTHON) paper/sec3_m3r/fig2_m3r_geometry.py
fig3:
	$(PYTHON) paper/sec4_single_manifold/fig3_jauc_mado.py
fig4:
	$(PYTHON) paper/sec4_single_manifold/fig4_pca_alignment.py
fig5:
	$(PYTHON) paper/sec4_single_manifold/fig5_ev_transfer.py
fig6:
	$(PYTHON) paper/sec5_decision_boundary/fig6_svm_curves.py
fig7:
	$(PYTHON) paper/sec5_decision_boundary/fig7_sv_fractions.py
fig8 table1:
	$(PYTHON) paper/sec6_unified/fig8_unified_repr.py

# Tier 2: analyses from the distance blocks (see README.md for runtimes)
sec2:
	$(PYTHON) paper/sec2_metrics/fig1_transport_plans.py --recompute
sec3: fig2
sec4:
	$(PYTHON) paper/sec4_single_manifold/run_intraclass.py
	$(PYTHON) paper/sec4_single_manifold/run_pca_structure.py
	$(MAKE) fig3 fig4 fig5
sec5:
	paper/sec5_decision_boundary/run_svm.sh
	$(MAKE) fig6 fig7
sec6:
	paper/sec6_unified/run_bdt.sh
	$(MAKE) fig8

clean-figures:
	rm -f figures/*.pdf figures/*.tex figures/*.csv
