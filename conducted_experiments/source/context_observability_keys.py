"""Audit signatures preserving every additional input of the frozen variants.

Category alone does not identify which side is the service. The complete
signature must include both role indicators; no classifier feature is changed.
"""
import numpy as np
from network_context_features import service_hints

def complete_signature(x,ports,features):
    h=service_hints(x,ports,features)
    return np.column_stack([h['category'],h['source_service_hint'],h['destination_service_hint']]).astype(np.uint8)

def full_observation_matrix(x,ports,features):
    return np.column_stack([x,complete_signature(x,ports,features)]).astype(np.float32)
