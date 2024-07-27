import torch

class AdaCoor(torch.optim.Optimizer):
    def __init__(self, params, eps=1e-8, *args, **kwargs):
        defaults = dict(eps=eps, lr=1)
        print("initializing AdaCoor with defaults", defaults)
        super(AdaCoor, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for i, group in enumerate(self.param_groups):
            with torch.no_grad():
                
                # Initialize epsilon as a tensor
                epsilon = torch.tensor([group['eps']], dtype=torch.bfloat16, device=next(iter(group['params'])).device)

                for p in group['params']:
                    if p.grad is None:
                        continue
                   
                    state = self.state[p]

                    # Initialize state variable for vt
                    if 'vt' not in state:
                        state['vt'] = torch.zeros_like(p.data, device=p.device).to(dtype=torch.bfloat16, device=p.device)

                    vt = state['vt']
                    vt.add_((epsilon * p.grad.data ** 2).to(dtype=torch.bfloat16, device=p.device))
                    
                    gt_hat = (epsilon * p.grad.data).to(dtype=torch.float32, device=p.device)

                    denom = vt.sqrt().add_(group['eps']).to(dtype=p.dtype, device=p.device)
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)

        return loss
