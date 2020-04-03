from train_level_1 import *
from train_level_3 import *


best_epoch = 0
best_acc1 = 0
best_loc1 = 0
best_loc1_epoch = 0
loc1_at_best_acc1 = 0
acc1_at_best_loc1 = 0
gtknown_at_best_acc1 = 0
gtknown_at_best_loc1 = 0
bbest_epoch = 0
writer = None

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_loc1, best_epoch, \
            loc1_at_best_acc1, acc1_at_best_loc1, \
            gtknown_at_best_acc1, gtknown_at_best_loc1
            
    global writer


    # print("now now now")
    args.gpu = gpu

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join('/data/wayne/train_log', args.name, TIMESTAMP)

    if args.gpu == 0:
        writer = SummaryWriter(logdir=log_folder)

    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    # Modified: change log.log to log.txt so it can be view on server
    Logger(os.path.join(log_folder, 'log.txt'))
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # print("here")

    if args.dataset == 'CUB':
        num_classes = 200
    elif args.dataset == 'ILSVRC':
        num_classes = 1000
    else:
        raise Exception("Not preferred dataset.")

    if args.arch == 'vgg16':
        model = vgg_eil.get_model(
            pretrained=True,
            progress=True,
            batch_norm=False,
            num_classes=num_classes,
            mode=args.mode
        )
    elif args.arch == 'google':
        model = drop_inception3.get_model(
            pretrained=True,
            progress=True,
            num_classes=num_classes,
            mode=args.mode
        )
    elif args.arch == 'resnet50_ADL':
        model = drop_resnet.resnet50(pretrained=True,
                                     progress=True,
                                     num_classes=num_classes,
                                     ADL_position=args.ADL_position,
                                     drop_rate=args.ADL_rate,
                                     drop_thr=args.ADL_thr)
    else:
        model = None
        print(args.arch)
        print("FAIL TO LOAD MODEL!!!")

    # print("The model arch")
    # for name,para in model.named_parameters():
    #     print(name)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    param_features = []
    param_classifiers = []

    if 'vgg' in args.arch:
        for name, parameter in model.named_parameters():
            if 'fc.' in name or '.classifier' in name:
                # print("name added to cls part",name)
                param_classifiers.append(parameter)
            else:
                # print("name added to feature part",name)
                param_features.append(parameter)
    elif 'google' in args.arch:
        for name, parameter in model.named_parameters():
            if 'fc.' in name or '.classifier' in name:
                print("name added to cls part", name)
                param_classifiers.append(parameter)
            else:
                # print("name added to feature part",name)
                param_features.append(parameter)
    elif args.arch.startswith('resnet'):
        for name, parameter in model.named_parameters():
            if 'layer4.' in name or 'fc.' in name:
                param_classifiers.append(parameter)
            else:
                param_features.append(parameter)
    else:
        raise Exception("Fail to recognize the architecture")

    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifiers, 'lr': args.lr * args.lr_ratio}],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nest)

    # optionally resume from a checkpoint
    if args.resume:
        model, optimizer = load_model(model, optimizer, args)

    cudnn.benchmark = True

    # CUB-200-2011
    train_loader, val_loader, train_sampler = data_loader(args)

    if args.cam_curve:
        evaluate(val_loader, model, log_folder, args)
        cam_curve(val_loader, model, writer, log_folder, args)
        return

    # test mode early return
    # TODO Update : codes needs a completely refresh emmm
    if args.evaluate:
        evaluate(val_loader, model, log_folder, args)
        return

    if args.gpu == 0:
        print("Batch Size per Tower: %d" % (args.batch_size))
        print(model)

    for epoch in range(args.start_epoch, args.epochs):
        if args.gpu == 0:
            print("===========================================================")
            print("Start Epoch %d ..." % (epoch+1))

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        val_acc1 = 0
        val_loss = 0
        val_gtloc = 0
        val_loc = 0



        # note  training start
        if args.mode == 'base':
            train_acc, train_loss, progress_train = \
                train(train_loader, model,  optimizer, epoch, log_folder, writer, args)
        elif args.mode == 'ACoL':
            train_acc, train_loss, progress_train, loss_raw, loss_erase = \
                train(train_loader, model,  optimizer, epoch, args)
        else:
            train_acc, train_loss, progress_train = \
                train(train_loader, model,  optimizer, epoch, args)

        if args.gpu == 0:
            progress_train.display(epoch+1)

        # evaluate on validation set

        # evaluate localization on validation set
        if args.task == 'wsol':
            val_acc1, val_acc5, val_loss, \
                val_gtloc, val_loc = evaluate_loc(
                    val_loader, model, epoch, log_folder, args)
        # no validate anymore
        elif args.task == 'cls':
            val_acc1, val_acc5, val_loss = validate(
                val_loader, model, epoch, args)


        # tensorboard
        if args.gpu == 0:
            writer.add_scalar(args.name + '/train_acc', train_acc, epoch)
            writer.add_scalar(args.name + '/train_cls_loss', train_loss, epoch)
            writer.add_scalar(args.name + '/val_cls_acc', val_acc1, epoch)
            writer.add_scalar(args.name + '/val_loss', val_loss, epoch)

            if args.task == 'wsol':
                writer.add_scalar(args.name + '/val_gt_loc', val_gtloc, epoch)
                writer.add_scalar(args.name + '/val_loc1', val_loc, epoch)

            if args.mode == 'ACoL':
                writer.add_scalar(
                    args.name + '/train_loss_raw', loss_raw, epoch)
                writer.add_scalar(
                    args.name + '/train_loss_erase', loss_erase, epoch)

        # remember best acc@1 and save checkpoint, this is top1 classification acc
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        if is_best:
            best_epoch = epoch + 1
            if args.task == 'wsol':
                loc1_at_best_acc1 = val_loc
                gtknown_at_best_acc1 = val_gtloc

        if args.task == 'wsol':
            # in case best loc,, Not using this.
            is_best_loc = val_loc > best_loc1
            best_loc1 = max(val_loc, best_loc1)
            if is_best_loc:
                best_loc1_epoch = epoch + 1
                acc1_at_best_loc1 = val_acc1
                gtknown_at_best_loc1 = val_gtloc

        if args.gpu == 0:
            if args.task == 'wsol':
                print("\nCurrent Best Epoch: %d" % (best_loc1_epoch))
                print("Top-1 GT-Known Localization Acc: %.3f \
                   \nTop-1 Localization Acc: %.3f\
                   \nTop-1 Classification Acc: %.3f" %
                      (gtknown_at_best_loc1, best_loc1, acc1_at_best_loc1))
                print("\nEpoch %d finished." % (epoch+1))
            else:
                print("\nCurrent Best Epoch: %d" % (best_epoch))
                print("Top-1 Classification Acc: %.3f" %
                      (best_acc1))
                print("\nEpoch %d finished." % (epoch+1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            saving_dir = os.path.join(log_folder)
            if args.task == 'wsol':
                save_criterion = is_best_loc
            else:
                save_criterion = is_best
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                # convert to torch tensor
                'best_acc1': torch.FloatTensor([best_acc1]),
                'optimizer': optimizer.state_dict(),
            }, save_criterion, saving_dir)


    # TODO : change the evaluate behavior, single GPU to perform evaluate
    if args.gpu == 0 and args.task.startswith('wsol'):
        args.resume = os.path.join(log_folder, 'model_best.pth.tar')
        if args.task.startswith('wsol_eval'):
            print("SET checkpoint.pth.tar")
            args.resume = os.path.join(log_folder, 'checkpoint.pth.tar')
        model, _ = load_model(model, optimizer, args)
        evaluate(val_loader, model, log_folder, args)
        cam_curve(val_loader, model,  writer, log_folder, args)
