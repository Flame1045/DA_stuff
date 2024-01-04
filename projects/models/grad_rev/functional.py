from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            print("self.alpha grad:",alpha.grad)
            grad_input = - alpha*grad_output
        # print("grad_ouput",grad_output)
        # print("grad_input",grad_input)
        return grad_input, None
revgrad = GradientReversal.apply