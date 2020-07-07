import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_seq_len = 17

def GenerateCaptions(features, captions, model):
    features = torch.tensor(features, device=device).float().unsqueeze(0)
    gen_caps = torch.tensor(captions[:, 0:1], device=device).long()
    for t in range(max_seq_len-1):
        output = model(features, gen_caps)
        gen_caps = torch.cat((gen_caps, output[:, -1:, :].argmax(axis=2)), axis=1)
    return gen_caps


def GenerateCaptionsWithBeamSearch(features, captions, model, beamSize=5):
    features = torch.tensor(features, device=device).float().unsqueeze(0)
    gen_caps = torch.tensor(captions[:, 0:1], device=device).long()
    candidates = [(gen_caps, 0)]
    for t in range(max_seq_len-1):
        next_candidates = []
        for c in range(len(candidates)):
            output = model(features, candidates[c][0])
            probs, words = torch.topk(output[:, -1:, :], beamSize)
            for i in range(beamSize):
                cap = torch.cat((candidates[c][0], words[:, :, i]), axis=1)
                score = candidates[c][1] - torch.log(probs[0, 0, i]).item()
                next_candidates.append((cap, score))
        ordered_candidates = sorted(next_candidates, key=lambda tup:tup[1])
        candidates = ordered_candidates[:beamSize]
    return candidates 


