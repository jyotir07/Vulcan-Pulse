"""The scenario grid the harness evaluates.

It spans every intervention type, magnitudes inside and well beyond what training episodes cover,
peak and off-peak timing, and two combined scenarios. Whether a scenario is in or out of the
models' range is decided from the data by the harness, not by these names.
"""


def _degradation(issuer: str, delta: float, method: str | None = "UPI", **timing) -> dict:
    return {
        "type": "issuer_degradation",
        "issuer": issuer,
        "method": method,
        "success_rate_delta": delta,
        **timing,
    }


def _outage(gateway: str, minutes: int, start_hour: int) -> dict:
    return {
        "type": "gateway_outage",
        "gateway": gateway,
        "start_hour": start_hour,
        "duration_minutes": minutes,
    }


def _routing(source: str, target: str, share: float, **timing) -> dict:
    return {
        "type": "routing_change",
        "source_gateway": source,
        "target_gateway": target,
        "traffic_percentage": share,
        **timing,
    }


def scenario_grid() -> list[tuple[str, list[dict]]]:
    grid: list[tuple[str, list[dict]]] = []

    evening = {"start_hour": 18, "duration_minutes": 120}
    for issuer in ("HDFC", "SBI", "AU Small Finance"):
        for delta in (-0.05, -0.10, -0.20, -0.40):
            grid.append(
                (
                    f"{issuer} UPI {delta * 100:+.0f}pp 18:00-20:00",
                    [_degradation(issuer, delta, **evening)],
                )
            )
    for delta in (-0.10, -0.30):
        grid.append(
            (f"HDFC all methods {delta * 100:+.0f}pp all day", [_degradation("HDFC", delta, None)])
        )

    for gateway in ("gateway_a", "gateway_b", "gateway_c", "gateway_d"):
        for minutes in (15, 60):
            grid.append((f"{gateway} outage {minutes}m at 19:00", [_outage(gateway, minutes, 19)]))

    for segment, delta in (
        (None, -0.3),
        (None, 0.3),
        (None, 0.8),
        (None, 1.2),
        ("UPI", 0.3),
        ("UPI", 0.8),
        ("CARD", 0.8),
    ):
        label = segment or "all"
        grid.append(
            (
                f"{label} traffic {delta * 100:+.0f}%",
                [{"type": "traffic_change", "segment": segment, "volume_delta": delta}],
            )
        )

    for source, target, share in (
        ("CARD", "UPI", 0.2),
        ("CARD", "UPI", 0.6),
        ("UPI", "CARD", 0.2),
        ("NETBANKING", "UPI", 0.5),
        ("UPI", "NETBANKING", 0.3),
    ):
        grid.append(
            (
                f"{share:.0%} {source} -> {target}",
                [{"type": "method_shift", "from": source, "to": target, "percentage": share}],
            )
        )

    for source, target, share in (
        ("gateway_a", "gateway_b", 0.25),
        ("gateway_a", "gateway_b", 0.75),
        ("gateway_a", "gateway_c", 0.5),
        ("gateway_b", "gateway_d", 0.5),
        ("gateway_c", "gateway_a", 0.5),
        ("gateway_d", "gateway_a", 0.9),
    ):
        grid.append((f"route {share:.0%} {source} -> {target}", [_routing(source, target, share)]))

    peak = {"start_hour": 19, "duration_minutes": 60}
    grid.append(
        (
            "gateway_a outage 60m, all its traffic rerouted to gateway_b",
            [_routing("gateway_a", "gateway_b", 1.0, **peak), _outage("gateway_a", 60, 19)],
        )
    )
    grid.append(
        (
            "HDFC UPI -15pp during a 50% traffic surge",
            [
                {"type": "traffic_change", "volume_delta": 0.5},
                _degradation("HDFC", -0.15),
            ],
        )
    )
    return grid
