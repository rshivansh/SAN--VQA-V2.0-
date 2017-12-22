require 'torch'
require 'nn'
require 'rnn'
require 'optim'
san = require 'misc/san'

opt = {
    -- data
    img_train_h5 = 'data/img_train.h5',
    img_test_h5 = 'data/img_test.h5',
    qa_h5 = 'data/qa.h5',
    params_json = 'data/params.json',

    -- model
    embedding_size = 200,
    rnn_size = 512,
    im_tr_size = 512,
    num_attention_layers = 2,
    common_embedding_size = 512,
    num_output = 1000,

    -- optim
    batch_size = 150,
    lr = 1e-3,
    decay_factor = 0.99997592083,
    max_iters = 150000,

    -- log
    log_path = 'logs/',
    log_file = 'train.log',

    -- misc
    gpu = 0,
    seed = 123,
    checkpoint_every = 10000,
    checkpoint_path = 'checkpoints/'
}

-- one-line argument parser. parses enviroment variables to override the defaults
-- from https://github.com/soumith/dcgan.torch/blob/master/main.lua#L25
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
paths.mkdir(opt.checkpoint_path) paths.mkdir(opt.log_path)
if opt.checkpoint_path == 'checkpoints/' then
    local cur_time = os.date('*t', os.time())
    opt.checkpoint_path = string.format('checkpoints/%d-%d-%d-%d:%d:%d/', cur_time.month, cur_time.day, cur_time.year, cur_time.hour, cur_time.min, cur_time.sec)
end
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- dataloader
dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, 'train')
collectgarbage()

print('Vocab size ' .. dataloader.vocab_size)

-- model
lm = nn.Sequential()
        :add(nn.LookupTableMaskZero(dataloader.vocab_size, opt.embedding_size))
        :add(nn.Dropout(0.5))
        :add(nn.SplitTable(1, 2))
        :add(nn.Sequencer(nn.FastLSTM(opt.embedding_size, opt.rnn_size):maskZero(1)))
        :add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size):maskZero(1)))
        :add(nn.SelectTable(-1))

model = nn.Sequential()
            :add(nn.ParallelTable()
                :add(nn.Identity())
                :add(lm))
            :add(san.n_attention_layer(opt))

-- criterion
criterion = nn.ClassNLLCriterion()

if opt.gpu >= 0 then
    print('Shipping to cuda')
    require 'cutorch'
    require 'cunn'
    model = model:cuda()
    criterion = criterion:cuda()
end

model_params, model_grad_params = model:getParameters()

state = {}
state.learningRate = opt.lr

-- closure to calculate accuracy over validation set
function feval_val(max_batches)
    count = 0
    n=loader.batch_size() ----- check what to write here
    if max_batches ~= nil then n = math.min(n, max_batches) end
    model:evaluate()
    
    for i = 1, n do
        
        -- load question batch, answer batch and image features batch
        batch = dataloader:next_batch(opt) -----check what to write here
        -- forward pass
        scores = model:forward({batch[1], batch[2]})
        
        -- count number of correct answers
        _, idx  = scores:max(2) -- check what to write here
        for j = 1, opt.batch_size do
            if idx[j][1] == a_batch[j] then
                count = count + 1
            end
        end
    end
    -- set to training mode once done with validation
    model:training()
    model:training()

    -- return accuracy
    return count / (n * opt.batch_size)

end 

-- closure to evaluate f and df/dx
function feval(x)
    if model_params ~= x then
        model_params:copy(x)
    end
    
    model_grad_params:zero()
    batch = dataloader:next_batch(opt)
    -- forward pass
    scores = model:forward({batch[1], batch[2]})
    loss = criterion:forward(scores, batch[3])
    -- backward pass
    dloss_dscores = criterion:backward(scores, batch[3])
    model:backward({batch[1], batch[2]}, dloss_dscores)

    model_grad_params:clamp(-10, 10)

    if running_avg_loss == nil then
        running_avg_loss = loss
    end
    running_avg_loss = running_avg_loss * 0.95 + loss * 0.05
    return loss, model_grad_params
end

-- training loop, with logging and checkpointing
logger = optim.Logger(opt.log_path .. opt.log_file)
for iter = 1, opt.max_iters do
    epoch = iter / (dataloader['ques']:size(1) / opt.batch_size)
    optim.adam(feval, model_params, state)
    if iter % 10 == 0 then
        print('training loss ' .. running_avg_loss, 'on epoch ' .. epoch, 'iter ' .. iter .. '/' .. opt.max_iters)
        logger:add{iter, epoch, running_avg_loss}
        collectgarbage()
    end
    
    state.learningRate = state.learningRate * opt.decay_factor
    
     -- Calculate validation accuracy and save model snapshot
    if iter % opt.checkpoint_every == 0 then
        print('Checkpointing. Calculating validation accuracy..')
        paths.mkdir(opt.checkpoint_path)
        local val_acc = feval_val()
        local save_file = string.format(opt.checkpoint_path .. 'iter_%d.t7', iter, epoch, val_acc)
        print('Saving checkpoint to ' .. save_file)
        torch.save(save_file, {model_params = model_params, opt = opt})
    end
end
