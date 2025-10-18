"""Risk Mitigation Efficiency (RME) = (baseline_impact - mitigated_impact) / mitigation_cost"""
def compute_rme(baseline_impact, mitigated_impact, mitigation_cost, eps=1e-9):
    impact_reduction = baseline_impact - mitigated_impact
    return (impact_reduction) / (mitigation_cost + eps)

if __name__ == '__main__':
    print('Example RME', compute_rme(100, 65, 10))
