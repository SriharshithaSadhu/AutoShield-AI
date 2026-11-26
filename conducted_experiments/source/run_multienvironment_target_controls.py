"""Extend the shared bank with exact-subset controls; reuse primary fitted caches."""
import run_multienvironment_training as experiment
from threadpoolctl import threadpool_limits
if __name__=='__main__':
 experiment.METHODS+=['single_unsw_matched50k','single_ton_matched150k']
 with threadpool_limits(limits=1):experiment.main()
