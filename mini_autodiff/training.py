def train_step(loss_fn, optimizer):
    optimizer.zero_grad()

    loss = loss_fn()

    loss.backward()

    optimizer.step()

    return loss

def train(loss_fn, optimizer, steps):
    losses = []

    for _ in range(steps):
        loss = train_step(loss_fn, optimizer)
        losses.append(float(loss.data))

    return losses