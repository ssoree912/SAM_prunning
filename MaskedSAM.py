import torch

class MaskedSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, v2=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(MaskedSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.v2 = v2
        
    def _masked_vec(self, p, grad, adaptive, mask):
        if grad is None or mask is None:
            return None
        wscale = p.abs() if adaptive else 1.0
        return (wscale * grad) * mask # mask가 0이면 e 는 0

    @torch.no_grad()
    def _grad_norm(self, mask_dict=None): #norm 계산 
        shared_device = self.param_groups[0]["params"][0].device
        grad_list = []

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: 
                    continue
                
                grad_to_use = None
                if self.v2 and p in self.state and "prev_grad" in self.state[p] and self.state[p]["prev_grad"] is not None:
                    grad_to_use = self.state[p]["prev_grad"]
                elif p.grad is not None:
                    grad_to_use = p.grad
                
                if grad_to_use is None: 
                    continue

                mask = mask_dict.get(p, None) if mask_dict else None
                v = self._masked_vec(p, grad_to_use, group["adaptive"], mask)
                
                if v is None:
                    continue
                    
                individual_norm = v.norm(p=2)
                grad_list.append(individual_norm.to(shared_device))

        if len(grad_list) == 0:
            return torch.tensor(0.0, device=shared_device)
        
        return torch.norm(torch.stack(grad_list), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False, mask_dict=None):
        grad_norm = self._grad_norm(mask_dict)
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if not p.requires_grad: 
                    continue
                
                if self.v2 and "prev_grad" in self.state[p]:
                    grad = self.state[p]["prev_grad"]
                else:
                    grad = p.grad
                
                if grad is None: 
                    continue

                mask = mask_dict.get(p, None) if mask_dict else None
                v = self._masked_vec(p, grad, group["adaptive"], mask)
                
                if v is None:
                    continue
                
                e_w = v * scale.to(p)
                p.add_(e_w)  # w ← w + ε
                self.state[p]["e_w"] = e_w
                
        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad: 
                    continue
                if p.grad is None: 
                    continue
                    
                if self.v2:
                    if torch.isinf(p.grad).any():
                        print(f"🔥🔥🔥 WARNING: Gradient contains INF for param shape: {p.shape} 🔥🔥🔥")
                        continue
                    elif torch.isnan(p.grad).any():
                        print(f"🔥🔥🔥 WARNING: Gradient contains NAN for param shape: {p.shape} 🔥🔥🔥")
                        continue
                    # (v2) 이번 스텝에서 계산된 grad를 prev_grad로 저장
                    self.state[p]["prev_grad"] = torch.clone(p.grad)
                    self.state[p]["prev_u"] = torch.clone(p)

                if "e_w" in self.state[p]:
                     # ε 복원
                    p.sub_(self.state[p]["e_w"]) # w ← w - ε
                    
                #복원 단계(w+ε → w). 복원 이후 바깥에서 base_optimizer.step()을 호출해 실제 업데이트를 수행
        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step()