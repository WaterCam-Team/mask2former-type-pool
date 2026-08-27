# Debug Guide for Prompt Learner

This guide explains how to use the debug tools to diagnose why the alignment loss is constant.

## Quick Start

### Option 1: Run Standalone Debug Script (Recommended)

The easiest way to check your setup is to run the standalone debug script:

```bash
python run_debug_check.py --config-file <your_config.yaml>
```

This will:
1. Build your model
2. Check if `prompt_ctx` parameters require gradients
3. Build the optimizer
4. Verify `prompt_ctx` parameters are in the optimizer
5. Run a dummy forward/backward pass to check gradient flow

### Option 2: Add Debug Calls to Training Script

You can also add debug calls directly in your training script. Edit `train_net_custom.py`:

#### Step 1: Import the debug functions

Add this import at the top of `train_net_custom.py`:

```python
from debug_prompt_learner import check_prompt_learner_setup, check_optimizer_params
```

#### Step 2: Check model setup after building

In the `build_optimizer` method, after building the optimizer, uncomment the debug lines:

```python
# ========== DEBUG: Check prompt_ctx parameters in optimizer ==========
from debug_prompt_learner import check_optimizer_params
print("\n[DEBUG] Checking optimizer parameters...")
check_optimizer_params(model, optimizer)
```

#### Step 3: Check model setup in main function

In the `main` function, after building the model:

```python
if args.eval_only:
    model = Trainer.build_model(cfg)
    # Add debug check here
    check_prompt_learner_setup(model)
    # ... rest of code
```

Or in the training path:

```python
trainer = Trainer(cfg)
# Add debug check before training
model = trainer.model
check_prompt_learner_setup(model)
optimizer = trainer.optimizer
check_optimizer_params(model, optimizer)
trainer.resume_or_load(resume=args.resume)
return trainer.train()
```

## What the Debug Output Shows

### 1. Prompt Learner Setup Check

```
[DEBUG] Checking Prompt Learner Setup
Found prompt_learner at: sem_seg_head.prompt_learner

1. Checking prompt_ctx parameters:
  prompt_ctx[0]:
    requires_grad: True  ← Should be True
    shape: torch.Size([3, 8, 512])
    mean: 0.000123
    std: 0.019876
```

**What to look for:**
- ✅ `requires_grad: True` - Parameters will receive gradients
- ❌ `requires_grad: False` - Parameters won't be updated (PROBLEM!)

### 2. Optimizer Check

```
[DEBUG] Checking Optimizer Parameters
Checking if prompt_ctx parameters are in optimizer:
  sem_seg_head.prompt_learner.prompt_ctx[0]: in_optimizer=True, requires_grad=True
```

**What to look for:**
- ✅ `in_optimizer=True` - Parameter is being optimized
- ❌ `in_optimizer=False` - Parameter is NOT in optimizer (PROBLEM!)

### 3. Gradient Flow Check (from training logs)

During training, you'll see debug output like:

```
[DEBUG Step 1] masktext_alignment_loss_mixneg_diffK
  text_clip: shape=torch.Size([2, 8, 512]), requires_grad=True
  mask_clip: shape=torch.Size([2, 100, 512]), requires_grad=True
  prompt_ctx[0]: requires_grad=True, mean=0.000123
  Final loss: 2.345678, requires_grad=True

[GRAD HOOK] prompt_ctx[0]: grad_norm=0.001234, grad_mean=0.000012
```

**What to look for:**
- ✅ `requires_grad=True` on text_clip and mask_clip
- ✅ `[GRAD HOOK]` messages showing gradient values
- ❌ `grad=None` or `grad_norm=0.0` - No gradients (PROBLEM!)

## Common Issues and Solutions

### Issue 1: `prompt_ctx` requires_grad=False

**Symptom:**
```
prompt_ctx[0]: requires_grad: False
```

**Solution:**
The parameters are not set to require gradients. Check that in `CLIPPromptLearner.__init__`, you have:
```python
for param in self.prompt_ctx:
    param.requires_grad = True
```

### Issue 2: `prompt_ctx` not in optimizer

**Symptom:**
```
prompt_ctx[0]: in_optimizer=False, requires_grad=True
```

**Solution:**
The optimizer building code is skipping these parameters. Check that `build_optimizer` is including all parameters with `requires_grad=True`. The issue might be in how parameters are collected.

### Issue 3: No gradients flowing

**Symptom:**
```
[GRAD HOOK] prompt_ctx[0]: grad=None!
```

**Solution:**
The computation graph is broken. Check:
1. Is the loss actually being computed? (check loss value)
2. Is `text_clip` connected to `prompt_ctx`? (check if `text_clip.requires_grad=True`)
3. Are there any `.detach()` calls breaking the graph?

### Issue 4: Constant loss values

**Symptom:**
Loss value doesn't change between iterations.

**Check:**
1. Are `prompt_ctx` parameter values changing? (compare mean/std between steps)
2. Are gradients non-zero? (check `[GRAD HOOK]` output)
3. Is the optimizer actually stepping? (check if parameters update)

## Interpreting Debug Output During Training

The debug code will print every 5 steps initially, then every 100 steps. Look for:

1. **Loss values changing:**
   ```
   Final loss: 2.345678  (step 1)
   Final loss: 2.123456  (step 2)  ← Should decrease
   ```

2. **Parameter values changing:**
   ```
   prompt_ctx[0]: mean=0.000123  (step 1)
   prompt_ctx[0]: mean=0.000145  (step 2)  ← Should change
   ```

3. **Gradients flowing:**
   ```
   [GRAD HOOK] prompt_ctx[0]: grad_norm=0.001234  ← Should be non-zero
   ```

## Disabling Debug Output

Once you've identified the issue, you can disable debug output by:

1. Commenting out the debug print statements
2. Or setting debug counters to only print on specific steps:
   ```python
   if self._debug_step_count <= 0:  # Disable by setting to 0
   ```

## Next Steps

After running the debug script:
1. Check if all checks pass (✓ marks)
2. If issues are found, fix them based on the solutions above
3. Re-run the debug script to verify fixes
4. Start training and monitor the debug output in logs

If you still see constant loss after fixing the issues, share the debug output and we can investigate further!

