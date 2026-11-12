"""
Theoretical energy accounting for spiking neural networks.

Energy model follows the standard neuromorphic cost split (Horowitz 2014, 45nm CMOS):
    - AC (accumulate) per active synaptic operation (spike): 0.9 pJ
    - MAC (multiply-accumulate) per dense floating-point operation: 4.6 pJ

In spike-driven layers, inputs are binary {0, 1}. The weight lookup on an active
spike is a pure addition (AC), and inactive spikes incur zero cost. Therefore:
    Active ACs = firing_rate × total_synaptic_connections × timesteps
"""


class EnergyMeter:
    def __init__(self, e_mac_pj: float = 4.6, e_ac_pj: float = 0.9):
        self.e_mac_pj = e_mac_pj
        self.e_ac_pj = e_ac_pj

    def calculate_sops(self, spikes, fan_out: int) -> int:
        """Total active synaptic operations = number of spikes * fan-out."""
        return int(spikes.sum().item()) * fan_out

    def estimate_energy_pj(self, sops: int, flops: int) -> float:
        """Theoretical energy in picojoules."""
        return sops * self.e_ac_pj + flops * self.e_mac_pj

    def estimate_snn_layer_energy_pj(
        self,
        total_ops: int,
        firing_rate: float,
        is_first_layer: bool = False,
    ) -> float:
        """Estimate energy for a single SNN layer with spike-driven accounting.

        Args:
            total_ops: Total synaptic connections (e.g. N * D_in * D_out * T).
            firing_rate: Empirical mean firing rate of the input spiking layer [0, 1].
            is_first_layer: If True, this layer receives continuous (analog) input
                (e.g. patch embedding on RGB pixels), so all ops are dense MACs.

        Returns:
            Estimated energy in picojoules.
        """
        if is_first_layer:
            # First layer receives continuous pixel values → dense MACs
            return total_ops * self.e_mac_pj
        else:
            # Spike-driven layer: only active spikes trigger additions (ACs)
            active_ops = int(total_ops * firing_rate)
            return active_ops * self.e_ac_pj
