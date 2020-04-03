from train_level_1 import *
from train_level_2 import *


def train(train_loader, model,  optimizer, epoch, log_folder, writer, args):

    batch_time = SumEpochMeter('Time', ':6.3f')
    data_time = SumEpochMeter('Data', ':6.3f')
    losses = AverageEpochMeter('Loss', ':.4e')
    top1 = AverageEpochMeter('Top-1 Classification Acc')
    top5 = AverageEpochMeter('Top-5 Classification Acc')
    learning_rate = AverageEpochMeter('Learning Rate:', ':.1e')
    if args.mode == 'ACoL':
        losses_erase = AverageEpochMeter('Loss Erase', ':.4e')
        losses_raw = AverageEpochMeter('Loss Raw', ':.4e')

    progress = ProgressEpochMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate, top1, top5],
        prefix="\nTraining Phase: ")

    for param_group in optimizer.param_groups:
        learning_rate.update(param_group['lr'])
        break

    # switch to train mode
    model.train()
    end = time.time()

    means = [.485, .456, .406]
    stds = [.229, .224, .225]
    selected = [4]  # , 2122, 4185, 4187, 4334, 5199]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)

    if args.gpu == 0:
        train_loader_wrapper = tqdm(enumerate(train_loader))
    else:
        train_loader_wrapper = enumerate(train_loader)

    for i, data in train_loader_wrapper:
        # if i==5:
        #     break
        (images, target, idx) = data

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images, target)
        # loss_cls = criterion(output, target)
        if args.mode == 'base':
            loss = model.module.get_loss(target)

        # todo

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if i < 5:
                # first five batch for every epoch
                image_ = images.clone().detach() * stds + means

                # so the image have to be normalized to [0,1]

                b, _, h, w = image_.shape
                # default as nearest interpolate
                image_ = F.interpolate(image_, (h, w))
                # take first 16 images
                image_set = vutils.make_grid(image_[:16], nrow=16)
                evaluate_img = image_.cpu().numpy().transpose(0, 2, 3, 1)
                # below code for temp
                # attention here is just from train data set
                gt_attention, _ = model.module.get_fused_cam(target)
                gt_attention = F.interpolate(gt_attention.unsqueeze(1), (h, w),
                                             mode='bilinear', align_corners=False)

                # blend heatmap with raw image, unerased feature map
                evaluate_cam = gt_attention.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
                evaluate_cam_holder = np.zeros(
                    (b, h, w, 3))  # change to batch size
                for k in range(b):
                    evaluate_cam_holder[k] = intensity_to_rgb(
                        evaluate_cam[k], normalize=True).astype('uint8')
                    evaluate_cam_holder[k] = evaluate_img[k] * 255 * \
                        0.6 + evaluate_cam_holder[k] * 0.4
                evaluate_cam_holder = torch.from_numpy(evaluate_cam_holder.transpose(
                    0, 3, 1, 2)).contiguous().cuda(args.gpu, non_blocking=True).float()
                evaluate_cam_holder_vutils = vutils.make_grid(evaluate_cam_holder[:16], nrow=16,
                                                              normalize=True, scale_each=True)
                if args.mode == 'base':
                    compare_set = torch.cat(
                        (image_set,
                            evaluate_cam_holder_vutils
                         ), dim=1)

                    # compare_set = torch.cat(
                    #     (image_set, gt_attention, evaluate_cam_holder, evaluate_unerased_pool4_holder), dim=1)

                if args.gpu == 0:
                    #  the image should be passed as a 3-dimension tensor of size [3, H, W].
                    # The three dimensions correspond to R, G, B channel of an image.
                    writer.add_image(
                        args.name+'/EIL_'+str(args.gpu), compare_set, epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg, progress

