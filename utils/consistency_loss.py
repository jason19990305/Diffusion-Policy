import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyDistillationLoss:
    """
    Consistency Distillation Loss for training a Consistency Policy from a Teacher model.
    """
    def __init__(self, teacher_model, solver, sigma_min=0.002, sigma_data=0.5):
        self.teacher_model = teacher_model
        self.solver = solver
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        
    def __call__(self, student_model, target_model, observations, actions, t, t_next, images=None):
        """
        actions: Ground truth actions from dataset.
        """
        device = actions.device
        
        # 1. Sample noise
        z = torch.randn_like(actions)
        
        # 2. Add noise to actions for t_next
        # (Assuming t_next is the larger noise level)
        x_next = self.solver.add_noise(actions, z, t_next)
        
        # 3. Predict next state using teacher model (1 step of ODE)
        # We need the solver's step function
        with torch.no_grad():
            # Get teacher's prediction
            # This depends on the solver type (e.g. DDIMScheduler)
            # x_t = teacher_step(x_next, t_next, t)
            
            # Simple DDIM-like step:
            # We can use the scheduler's step function
            # But the scheduler's step usually takes a model output
            teacher_noise = self.teacher_model(observations, t_next, x_next, images=images)
            # DDIMScheduler.step gives both x_t (prev_sample) and estimated x_0 (pred_original_sample)
            out = self.solver.step(teacher_noise, t_next[0], x_next)
            x_t = out.prev_sample
            x_0_teacher = out.pred_original_sample
            
        # 4. Student prediction at t_next
        student_pred = student_model(observations, t_next.float(), x_next, images=images)
        
        # 5. Target prediction at t using EMA model
        with torch.no_grad():
            target_pred = target_model(observations, t.float(), x_t, images=images)
            
        # 6. Triple Loss: Self + Teacher + Ground Truth
        # Use Huber Loss (Smooth L1) for GT and Teacher to reduce sensitivity to outliers
        # (This is standard practice for Consistency Policy)
        loss_self = F.mse_loss(student_pred, target_pred)
        loss_teacher = F.huber_loss(student_pred, x_0_teacher, delta=1.0)
        loss_gt = F.huber_loss(student_pred, actions, delta=1.0)
        

        # Weighted combination
        # The official paper relies heavily on Ground Truth (DSM) and Self-Consistency.
        loss = 0.3 * loss_self + 0.2 * loss_teacher + 0.5 * loss_gt
        
        return loss
