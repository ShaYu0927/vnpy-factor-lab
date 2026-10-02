from .base import MathBase

class ProbabilityMath(MathBase):
    """概率相关公式。"""

    @classmethod
    def conditional(cls, joint: float, condition: float) -> float:
        """P(A | B) = P(A ∩ B) / P(B)"""
        if not cls.is_valid(joint) or not cls.is_valid(condition):
            raise ValueError("概率必须是有限数")

        joint     = float(joint)
        condition = float(condition)

        if not 0 <= joint <= condition <= 1:
            raise ValueError("必须满足 0 <= joint <= condition <= 1")
        if condition == 0:
            raise ValueError("条件事件概率为 0, 条件概率未定义")

        return joint / condition