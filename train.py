from data import data, test_data, VOCAB_SIZE

import torch
from random import shuffle
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

LEARNING_RATE = 0.001
MAX_EPOCHS = 600
BATCH_SIZE = 100
PAD_LENGTH = 320
LOG_INTERVAL = 10
TEST_SIZE = 100
TERMINAL_ACC = 54

def pad(cont):
    if len(cont) > PAD_LENGTH:
        return cont[:PAD_LENGTH]
    return cont + [VOCAB_SIZE] * (PAD_LENGTH - len(cont))

def padlen(cont):
    return min(len(cont), PAD_LENGTH)

ptr = len(data)
def next_batch():
    global ptr
    if ptr + BATCH_SIZE > len(data):
        # Requires shuffle and start over
        shuffle(data)
        ptr = 0

    result = data[ptr:ptr+BATCH_SIZE]

    # Sort by length in decreasing order. This is needed for RNN packing/padding
    result.sort(key = lambda s: len(s[1]), reverse=True)

    ptr += BATCH_SIZE
    return result

def train(model, save_path, pass_length = False, max_epochs = MAX_EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    for epoch in range(0, max_epochs):
        batch = next_batch()

        # Manually pads
        target = torch.tensor([row[0] for row in batch], dtype=torch.double).cuda()
        words = torch.tensor([pad(row[1]) for row in batch], dtype=torch.long).cuda()
        lengths = torch.tensor([padlen(row[1]) for row in batch], dtype = torch.long).cuda()

        optimizer.zero_grad()
        if pass_length:
            output = model(words, lengths)
        else:
            output = model(words)

        # Softmax is implied here
        loss = F.cross_entropy(output, torch.max(target, 1)[1])

        if (epoch + 1) % LOG_INTERVAL == 0:
            # Log train-acc
            corrects = (torch.max(output, 1)[1].data == torch.max(target, 1)[1].data).sum().item()
            acc = 100.0 * corrects / BATCH_SIZE

            with torch.no_grad():
                # Log test-acc, run all at once
                test_corrects = 0
                buckets = [0,0,0,0,0,0,0,0]
                target_buckets = [0,0,0,0,0,0,0,0]
                results = []
                targets = []
                for batch_ptr in range(0, len(test_data), TEST_SIZE):
                    batch = test_data[batch_ptr : batch_ptr + TEST_SIZE]
                    test_target = torch.tensor([row[0] for row in batch], dtype=torch.double).cuda()
                    test_words = torch.tensor([pad(row[1]) for row in batch], dtype=torch.long).cuda()
                    test_lengths = torch.tensor([padlen(row[1]) for row in batch], dtype = torch.long).cuda()
                    if pass_length:
                        test_output = model(test_words, test_lengths)
                    else:
                        test_output = model(test_words)

                    results += torch.max(test_output, 1)[1].tolist()
                    targets += torch.max(test_target, 1)[1].tolist()

                    test_corrects += (torch.max(test_output, 1)[1].data == torch.max(test_target, 1)[1].data).sum().item()

                for i in results:
                    buckets[i] += 1
                for i in targets:
                    target_buckets[i] += 1

                test_acc = 100.0 * test_corrects / len(test_data)
                print("Epoch: " + str(epoch), flush=True)
                print('Train: {:.6f}%'.format(acc))
                print('Loss:  {:.6f}'.format(loss.item()))
                print('Acc:   {:.6f}%'.format(test_acc), flush=True)
                print('Buck:  ' + str(buckets), flush=True)
                print('TBuck: ' + str(target_buckets), flush=True)
                print('F(Macro/Micro/Weighted/All):')
                print('  ' + str(f1_score(targets, results, average='macro')))
                print('  ' + str(f1_score(targets, results, average='micro')))
                print('  ' + str(f1_score(targets, results, average='weighted')))
                print('  ' + ', '.join(map(lambda l: str(l.item()), f1_score(targets, results, average=None))), flush=True)
                print('Pearson Cor:')
                print('  ' + str(pearsonr(targets, results)))

                if test_acc > TERMINAL_ACC:
                    print('Early terminate')
                    break

        loss.backward()
        optimizer.step()

    print('Final test')
    final_corrects = 0
    with torch.no_grad():
        for batch_ptr in range(0, len(test_data), TEST_SIZE):
            batch = test_data[batch_ptr : batch_ptr + TEST_SIZE]
            test_target = torch.tensor([row[0] for row in batch], dtype=torch.double).cuda()
            test_words = torch.tensor([pad(row[1]) for row in batch], dtype=torch.long).cuda()
            test_lengths = torch.tensor([padlen(row[1]) for row in batch], dtype = torch.long).cuda()
            if pass_length:
                test_output = model(test_words, test_lengths)
            else:
                test_output = model(test_words)
            final_corrects += (torch.max(test_output, 1)[1].data == torch.max(test_target, 1)[1].data).sum().item()

    print('Corrects: {}'.format(final_corrects))
    print('Total Acc: {:.6f}%'.format(100.0 * final_corrects / len(test_data)))

    print("Saving...")
    torch.save(model.state_dict(), save_path)
    print("Model saved to "+save_path, flush=True)
