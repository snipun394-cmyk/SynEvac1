# Physical CCTV Access Checklist

Practical, fill-in-the-blanks checklist for the day physical college CCTV/NVR access becomes available. Scope: **one test camera only** — see `docs/architecture/cctv_integration_readiness.md` §18.2 for why the first milestone must stay narrowly scoped to `RTSPFrameSource` → `CameraFrame` alone, with no detection or identity resolution attempted yet.

**Never write real credentials into this file, into any file committed to the repository, or into any project `.syn` file.** Passwords belong in `credential_store` (`credential_store/local_file_store.py`, `Path.home() / ".synevac" / "credentials.json"`, outside the repo) only, captured via the Property Panel exactly as `docs/architecture/cctv_integration_readiness.md` §7 describes. Where this checklist needs to record *how* a credential is stored, record a reference (e.g. "stored under camera's own id"), never the value itself.

## Camera Identity

- [ ] Digital Twin `camera_id` (the exact `Camera.id` already registered in the Building Designer / Camera Manager Panel — copy it, don't retype it)
- [ ] Camera display name
- [ ] Floor
- [ ] Zone(s) — ideally exactly one (see readiness doc §4: a camera spanning multiple zones can't honestly localize a detection without calibration this codebase doesn't yet do)
- [ ] Physical camera location (room / hallway / stairwell — human-readable description for whoever is on-site during setup)

## Network

- [ ] Camera IP address
- [ ] NVR IP address, if applicable
- [ ] Is direct camera access allowed, or must it go through the NVR?
- [ ] Network/VLAN restrictions (is the SynEvac machine on the same network segment? firewall rules needed?)
- [ ] Does the camera/NVR allow multiple simultaneous RTSP clients? (SynEvac reading the stream must not disrupt any existing recording/monitoring system already attached to it)

## Stream

- [ ] RTSP URL or stream path
- [ ] Main stream vs. substream (which one SynEvac should use — substream may be preferable for lower bandwidth during early testing)
- [ ] Codec (H.264 / H.265 / other)
- [ ] Resolution
- [ ] FPS
- [ ] Transport requirements (TCP / UDP, or whatever the camera/NVR documentation specifies)

## Authentication

- [ ] Username
- [ ] Password — **do not write the value here.** Record only that it exists and where it will be stored (`credential_store` under the camera's own `credential_ref`, per the Property Panel flow).
- [ ] Authentication type (Basic / Digest / other, if known)

## Device

- [ ] Manufacturer
- [ ] Model
- [ ] Firmware version, if available
- [ ] ONVIF support, if known (not assumed — see readiness doc §18, no ONVIF code exists or is planned until this is confirmed)

## Tests (run outside SynEvac first)

- [ ] Can VLC (or an equivalent RTSP client) open the stream?
- [ ] Can OpenCV/FFmpeg read frames from it? (a standalone script, not SynEvac — SynEvac has no frame-decoding library wired in yet by design, see readiness doc §14/§18.1)
- [ ] Average frame latency
- [ ] Dropped frames observed
- [ ] Reconnect behavior (what happens if the stream is interrupted — does the camera/NVR recover on its own, or does the client need to re-initiate?)

## Handoff to SynEvac integration (after the above is filled in)

Once every box above is checked for one camera, proceed per `docs/architecture/cctv_integration_readiness.md` §13 (steps 1-7) and §18.2 (Milestone A): match this camera to its Digital Twin `Camera` asset, configure `ConnectionInfo` in the Property Panel, save the password once (captured into `credential_store`), and only then begin writing `RTSPFrameSource` against the confirmed RTSP URL/codec/transport recorded here.
