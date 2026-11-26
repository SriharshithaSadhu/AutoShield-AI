"""Training-audit-motivated resolution ablation; independent output folder."""
import run_multienvironment_training as e
from threadpoolctl import threadpool_limits
if __name__=='__main__':
 e.O=e.R/'results/multienvironment_2026-11-21/harmonization'
 e.METHODS=['single_ton','single_unsw','pooled','environment_balanced'];e.REPRESENTATIONS=['harmonized14']
 with threadpool_limits(limits=1):e.main()
