import ast

class FlopsEstimator(ast.NodeVisitor):
    def __init__(self):
        # 初始化FLOPs为0
        self.flops = 0

    def visit_Mult(self, node):
        # 当遇到乘法操作时，FLOPs加1
        self.flops += 1
        self.generic_visit(node)

    def visit_Add(self, node):
        # 当遇到加法操作时，FLOPs加1
        self.flops += 1
        self.generic_visit(node)

    def visit_Sub(self, node):
        # 当遇到减法操作时，FLOPs加1
        self.flops += 1
        self.generic_visit(node)

    def visit_Div(self, node):
        # 当遇到除法操作时，由于除法比乘法复杂，我们用一个近似值。
        # 这里我们简单地使用近似值为10次乘法。
        self.flops += 10
        self.generic_visit(node)

    # ... 您可以为其他操作添加更多的visit方法 ...

def estimate_flops(code):
    # 解析代码
    tree = ast.parse(code)
    # 创建估算器实例
    estimator = FlopsEstimator()
    # 遍历抽象语法树
    estimator.visit(tree)
    # 返回估计的FLOPs
    return estimator.flops

if __name__ == '__main__':
    