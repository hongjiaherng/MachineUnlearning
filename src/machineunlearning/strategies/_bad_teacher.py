"""Bad Teacher / blindspot unlearning helpers.

Paper: Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an
       Incompetent Teacher — https://arxiv.org/abs/2205.08096
Source: https://github.com/vikram2000b/bad-teaching-unlearning
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from machineunlearning.data.dataset import UnlearningDataset


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample, label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_data_loader,
    optimizer,
    device,
    KL_temperature,
):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    epochs=10,
    optimizer="adam",
    lr=0.01,
    batch_size=256,
    device="cuda",
    KL_temperature=1,
):
    unlearning_data = UnlearningDataset(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )
        print(f"Epoch {epoch + 1} Unlearning Loss {loss}")
