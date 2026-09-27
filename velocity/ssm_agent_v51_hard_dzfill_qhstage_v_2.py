"""Compatibility import for the sampled-Q_h core.

The implementation lives in ssm_agent_v51_hard_dzfill_qhstage.py. The velocity
learner LPPSAgent uses its critic updates and supplies its own posterior-SNIS
actor update.
"""

from ssm_agent_v51_hard_dzfill_qhstage import SSMOnlineAgent

