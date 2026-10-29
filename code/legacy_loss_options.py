"""Compatibility checks for removed loss-weight command-line options."""
import argparse
import warnings


class RemovedLossWeight(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            value = float(values)
        except (TypeError, ValueError):
            parser.error(f"{option_string} requires a numeric value")
        if value != 0.0:
            parser.error(f"{option_string} was removed; only 0 is accepted for legacy commands. "
                         "Remove this option to use the current training objective.")
        warnings.warn(f"{option_string} is obsolete; accepted 0 for compatibility. "
                      "This loss is not used.", FutureWarning, stacklevel=2)
        setattr(namespace, self.dest, 0.0)
