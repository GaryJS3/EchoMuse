# Other changes

These ancillary fixes are separate from the fork's main purpose: the external
voice backend described in [External voice](external-voice.md).

## Provisioning storage check

The Install EchoMuse step now checks available space on `/data` before uploading
the server binary. A full filesystem previously left the upload stuck or failed
with `Socket closed`. The wizard reports available and required space, allowing
the operator to free storage in TWRP before retrying. It does not delete device
files automatically.

The estimate reserves two binary copies for staging and installation, plus
20 MiB for wake word assets and headroom. Unrecognised `df` output stops the
install with the device's output instead of guessing available space. Tests cover
the wrapped BusyBox output from the failing device, zero available blocks,
available space, and missing or unsupported output.
