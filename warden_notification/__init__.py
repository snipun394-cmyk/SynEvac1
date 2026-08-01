# Warden Notification -- notifies a human warden/responder of a zone
# that needs attention, tracks acknowledgement/completion. Same
# submit -> approve -> dispatch -> confirm/fail shape as
# building_control/, voice_evacuation/, dynamic_signage/. This is NOT
# real hardware/network dispatch: no SMS/push/email/webhook transport
# exists anywhere in this codebase -- SimulationWardenNotificationProvider
# is pure bookkeeping, exactly mirroring SimulationControlProvider's own
# "no backing physics" honesty for its state-only systems.
