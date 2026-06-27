"""CD(CL) viscous drag polar definitions for the Geometry API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CdclPolar:
    """Piecewise-quadratic CD(CL) polar used by AVL's ``CDCL`` keyword.

    The polar is defined by three anchor points, ordered by lift coefficient:

    * **Negative stall** — ``(cl_neg, cd_neg)``
    * **Minimum drag** — ``(cl_min, cd_min)``
    * **Positive stall** — ``(cl_pos, cd_pos)``

    AVL requires ``cl_neg < cl_min < cl_pos``. Drag coefficients are expected
    to be non-negative; ``cd_min > 0`` enables viscous strip drag in the solver.

    Parameters
    ----------
    cl_neg, cd_neg:
        Lift and drag at the negative-stall anchor.
    cl_min, cd_min:
        Lift and drag at the minimum-drag point.
    cl_pos, cd_pos:
        Lift and drag at the positive-stall anchor.
    """

    cl_neg: float
    cd_neg: float
    cl_min: float
    cd_min: float
    cl_pos: float
    cd_pos: float

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_points(
        cls,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        point_c: tuple[float, float],
    ) -> CdclPolar:
        """Build a polar from three ``(CL, CD)`` pairs in any order.

        Points are sorted by ``CL`` before validation.
        """
        ordered = sorted((point_a, point_b, point_c), key=lambda pair: pair[0])
        (cl_neg, cd_neg), (cl_min, cd_min), (cl_pos, cd_pos) = ordered
        return cls(
            cl_neg=cl_neg,
            cd_neg=cd_neg,
            cl_min=cl_min,
            cd_min=cd_min,
            cl_pos=cl_pos,
            cd_pos=cd_pos,
        )

    @classmethod
    def from_list(cls, values: list[float]) -> CdclPolar:
        """Build a polar from AVL's canonical six-value ``CDCL`` array.

        Expected order: ``[cl_neg, cd_neg, cl_min, cd_min, cl_pos, cd_pos]``.
        """
        if len(values) < 6:
            raise ValueError(f"CDCL polar requires 6 values, got {len(values)}")
        return cls(
            cl_neg=float(values[0]),
            cd_neg=float(values[1]),
            cl_min=float(values[2]),
            cd_min=float(values[3]),
            cl_pos=float(values[4]),
            cd_pos=float(values[5]),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if lift or drag values are invalid."""
        if self.cl_neg >= self.cl_min:
            raise ValueError(
                f"cl_neg ({self.cl_neg}) must be less than cl_min ({self.cl_min})"
            )
        if self.cl_min >= self.cl_pos:
            raise ValueError(
                f"cl_min ({self.cl_min}) must be less than cl_pos ({self.cl_pos})"
            )

        for name, cd in (
            ("cd_neg", self.cd_neg),
            ("cd_min", self.cd_min),
            ("cd_pos", self.cd_pos),
        ):
            if cd < 0.0:
                raise ValueError(f"{name} ({cd}) must be non-negative")

    @property
    def is_active(self) -> bool:
        """Return whether this polar enables viscous strip drag in the solver."""
        return self.cd_min != 0.0

    def as_list(self) -> list[float]:
        """Return the six-value array consumed by the AVL parser and solver."""
        return [
            self.cl_neg,
            self.cd_neg,
            self.cl_min,
            self.cd_min,
            self.cl_pos,
            self.cd_pos,
        ]
