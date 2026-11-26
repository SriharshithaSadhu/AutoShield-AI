"""Identical calibration controls on the resolution-harmonization ablation."""
import shutil
import run_multienvironment_calibration_control as b
import run_multienvironment_margin_control as c
from threadpoolctl import threadpool_limits
if __name__=='__main__':
 root=b.O;b.O=root/'harmonization';b.B=b.O/'network_calibration';c.O=b.O;c.B=b.B;c.C=c.O/'calibration_margin'
 shutil.copyfile(root/'calibration_margin_protocol.json',b.O/'calibration_margin_protocol.json')
 with threadpool_limits(limits=1):b.main();c.main()
