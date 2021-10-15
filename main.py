"""
The code demonstartes concepts presented in: Demisse G, et al. "Approximately covariant convolutional networks", 2021. 
"""

from __future__ import absolute_import, division
import trainlogic  as tl 

if __name__ == "__main__": 
    tl.parse()
    tl.verbose()
    tl.prepare_data()
    tl.main() 