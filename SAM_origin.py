


import torch

class SAM_origin(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, v2=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_origin, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.v2 = v2

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:

                if not p.requires_grad: continue
                if self.v2 and "prev_grad" in self.state[p]:

                    grad = self.state[p]["prev_grad"]
                    prev_p = self.state[p]["prev_grad"] #여기 원래 그래디언트가 아니라... 
                else:
                    grad = p.grad
                    prev_p = p
                if grad is None: 
                    continue

                e_w = (torch.pow(prev_p, 2) if group["adaptive"] else 1.0) * grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

                
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue
                if p.grad is None: continue
                if self.v2:
                    if torch.isinf(p.grad).any():
                        print(f"🔥🔥🔥 WARNING: Gradient contains INF for param shape: {p.shape} 🔥🔥🔥")
                        continue
                    elif torch.isnan(p.grad).any():
                        print(f"🔥🔥🔥 WARNING: Gradient contains NAN for param shape: {p.shape} 🔥🔥🔥")
                        continue
                        
                    self.state[p]["prev_grad"] = torch.clone(p.grad)
                    self.state[p]["prev_u"] = torch.clone(p)
                if p.grad is None: continue
                if "e_w" in self.state[p]:

                    p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        #self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step()


    def _grad_norm(self):
        # 오직 p.grad를 사용해 norm을 계산하도록 단순화
        shared_device = self.param_groups[0]["params"][0].device
        grad_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: continue # 필요 없으면 다음 파라미터로 넘어갑니다.
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad
                if grad is None: continue 

                
                adaptive_scale_factor = torch.pow(p, 2) if group["adaptive"] else 1.0
                term_to_calculate_norm_of = adaptive_scale_factor * p.grad
                
                individual_norm = term_to_calculate_norm_of.norm(p=2)
                grad_list.append(individual_norm.to(shared_device))

        if len(grad_list) == 0:
            return torch.tensor(0.0, device=shared_device)
        norm = torch.norm(torch.stack(grad_list), p=2)
        return norm