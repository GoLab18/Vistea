"""Edge scroll detector (EdgeScrollDetector) tests, including grace."""

from core.gestures import EdgeScrollDetector


def test_glance_shorter_than_entry_delay_never_scrolls():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.4)
    ticks = sum(det.update(0.05, t * 0.05) for t in range(7)) # 0.30 s in zone
    ticks += det.update(0.5, 0.40) # leaves
    assert ticks == 0


def test_scroll_arms_after_entry_delay_and_ticks_at_interval():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.4, tick_interval=0.12)
    total = 0
    t = 0.0
    while t <= 2.0:
        total += det.update(0.05, t) # top zone -> +1 per tick
        t += 0.02

    # ~(2.0 - 0.4) / 0.12 + 1 first tick = ~14; allow small boundary slack
    assert 12 <= total <= 15
    assert det.active


def test_bottom_zone_scrolls_down():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.1, tick_interval=0.1)
    total = 0
    t = 0.0
    while t <= 1.0:
        total += det.update(0.95, t)
        t += 0.02
    assert total < 0


def test_hysteresis_boundary_jitter_does_not_toggle():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.1, tick_interval=0.1, hysteresis=0.05)
    t = 0.0
    while t <= 0.3: # arm inside the zone
        det.update(0.10, t)
        t += 0.02
    assert det.active

    # jitter just OUTSIDE the entry boundary but INSIDE zone+hysteresis
    ticks = 0
    while t <= 1.0:
        ticks += det.update(0.15, t)  # 0.12 < 0.15 < 0.17
        t += 0.02
    assert det.active and ticks > 0 # kept scrolling through the jitter
    det.update(0.30, t) # clearly out
    assert not det.active


def test_lost_gaze_resets_gesture():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.1, grace=0.3)
    t = 0.0
    while t <= 0.3:
        det.update(0.05, t)
        t += 0.02
    assert det.active
    det.update(None, 0.35) # fade begins
    det.update(None, 0.75) # fade exceeds grace -> full reset
    assert not det.active and det.current_zone == 0


def test_brief_signal_loss_during_active_scroll_freezes_without_burst():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.1, tick_interval=0.1, grace=0.3)
    t = 0.0
    while t <= 0.3: # arm and activate
        t_call = t
        det.update(0.05, t_call)
        t += 0.02
    assert det.active

    ticks_during_fade = det.update(None, t_call + 0.05) # shorter than grace
    assert ticks_during_fade == 0
    assert det.active # state preserved, not reset

    # signal returns right after the fade began: the fade duration must not
    # count toward tick_interval, so this single frame yields no backlog.
    ticks_after = det.update(0.05, t_call + 0.06)
    assert ticks_after in (0, 1)
    assert det.active


def test_brief_signal_loss_during_arming_freezes_entry_delay():
    det = EdgeScrollDetector(zone=0.12, entry_delay=0.4, grace=0.3)
    t = 0.0
    while t <= 0.2: # inside the zone, not yet armed
        det.update(0.05, t)
        t += 0.02
    assert not det.active

    det.update(None, t + 0.1) # fade shorter than grace
    assert not det.active

    # Signal returns. frozen time should not count toward entry_delay, so a
    # single frame right after the fade must not be enough to arm yet
    ticks = det.update(0.05, t + 0.1)
    assert ticks == 0
    assert not det.active
