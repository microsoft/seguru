use std::fs::File;
use std::path::Path;

use cudarc::cublas::sys as cublas_sys;
use gpu_host::{GpuCtxGuard, GpuCtxSpace, PinnedHostBox, TensorViewMut};
use log::info;
use memmap2::Mmap;

pub(crate) mod dataloader;

#[macro_use]
pub mod kernels;
pub(crate) mod params;

use dataloader::parse_header_data;
use kernels::*;
use params::{ActivationTensors, GPT2Config, GradActTensors, ParameterTensors};

pub struct GPT2<'ctx, 'g, NS: GpuCtxSpace> {
    pub ctx: &'g GpuCtxGuard<'ctx, 'g, NS>,
    pub module: &'g gpu_host::GpuModule<NS>,
    /// Model configuration.
    pub config: GPT2Config,

    /// Total number of parameters.
    pub num_parameters: usize,

    /// The weights (parameters) of the model.
    pub params: ParameterTensors<'g>,

    /// Gradients of the weights.
    pub grads: Option<ParameterTensors<'g>>,

    /// Buffer for the AdamW optimizer.
    pub m_memory: Option<TensorViewMut<'g, [f32]>>,

    /// Buffer for the AdamW optimizer.
    pub v_memory: Option<TensorViewMut<'g, [f32]>>,

    /// The activations of the model.
    pub acts: Option<ActivationTensors<'g>>,

    /// Total number of activations.
    pub num_activations: usize,

    /// Gradients of the activations.
    pub grads_acts: Option<GradActTensors<'g>>,

    /// The batch size (B) of the current forward pass
    pub batch_size: usize,

    /// The sequence length (T) of the current forward pass
    pub seq_len: usize,

    /// The input tokens for the current forward pass
    pub inputs: Option<TensorViewMut<'g, [i32]>>,

    /// The target tokens for the current forward pass
    pub targets: Option<TensorViewMut<'g, [i32]>>,

    /// After a forward pass with targets, will be populated with the mean loss
    pub mean_loss: f32,

    pub cpu_losses: Option<PinnedHostBox<'g, [f32]>>,
}

impl<'ctx, 'g, NS: GpuCtxSpace> GPT2<'ctx, 'g, NS> {
    /// Creates a new GPT-2 model instance from a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to the checkpoint file containing model parameters and configuration.
    ///
    /// # Returns
    ///
    /// A new `GPT2` model instance.
    pub fn new(
        ctx: &'g GpuCtxGuard<'ctx, 'g, NS>,
        m: &'g gpu_host::GpuModule<NS>,
        checkpoint_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Read model from a checkpoint file
        let model_file = File::open(checkpoint_path).unwrap_or_else(|_| {
            panic!("Error opening model file: {:?}", checkpoint_path);
        });
        let model_data = unsafe { Mmap::map(&model_file)? };
        let (model_header, cpu_params) = parse_header_data(&model_data, 20240326, 3);
        if model_header[1] != 3 {
            panic!("Bad version in model file");
        }

        // Read in hyperparameters
        let max_seq_len = model_header[2] as usize;
        let vocab_size = model_header[3] as usize;
        let num_layers = model_header[4] as usize;
        let num_heads = model_header[5] as usize;
        let channels = model_header[6] as usize;
        let padded_vocab_size = model_header[7] as usize;

        let config = GPT2Config {
            max_seq_len,
            vocab_size,
            padded_vocab_size,
            num_layers,
            num_heads,
            channels,
        };
        println!("[GPT-2]");
        println!("max_seq_len: {}", max_seq_len);
        println!("vocab_size: {}", vocab_size);
        println!("padded_vocab_size: {}", padded_vocab_size);
        println!("num_layers: {}", num_layers);
        println!("num_heads: {}", num_heads);
        println!("channels: {}", channels);

        let param_sizes = config.get_params_sizes();
        // Count the number of parameters
        let num_parameters: usize = param_sizes.iter().sum();
        println!("num_parameters: {}", num_parameters);
        let cpu_params = &cpu_params[0..num_parameters];
        let params = ParameterTensors::new(ctx, param_sizes, cpu_params);

        Ok(GPT2 {
            config,
            ctx,
            module: m,
            num_parameters,
            params,
            grads: None,
            m_memory: None,
            v_memory: None,
            acts: None,
            num_activations: 0,
            grads_acts: None,
            batch_size: 0,
            seq_len: 0,
            inputs: None,
            targets: None,
            mean_loss: -1.0,
            cpu_losses: None,
        })
    }

    pub fn forward(
        &mut self,
        cublas_handle: cublas_sys::cublasHandle_t,
        cpu_inputs: &[i32],
        cpu_targets: &[i32],
        batch_size: usize,
        seq: usize,
    ) {
        let ctx = self.ctx;
        let ch = self.config.channels;
        let nh = self.config.num_heads;
        let m = self.module;
        let bsize = batch_size;
        let vocab = self.config.vocab_size;
        let pad_vocab = self.config.padded_vocab_size;
        assert!(cpu_inputs.len() >= bsize * seq);
        cpu_inputs.iter().for_each(|&x| assert!(0 <= x && (x as usize) < vocab, "{}", x));
        cpu_targets.iter().for_each(|&x| assert!(0 <= x && (x as usize) < vocab, "{}", x));

        // allocate space for all the activations if needed (done here, lazily)
        if self.acts.is_none() {
            let act_sizes = self.config.get_act_sizes(bsize, seq);
            self.num_activations = act_sizes.iter().sum();
            self.batch_size = bsize;
            self.seq_len = seq;
            self.acts =
                Some(ActivationTensors::new(ctx, act_sizes, &vec![0.0f32; self.num_activations]));
            println!(
                "allocated {} MiB for activations",
                (self.num_activations * std::mem::size_of::<f32>()) / (1024 * 1024)
            );
        }

        // also create memory for caching inputs and targets
        if self.inputs.is_none() {
            self.inputs = Some(ctx.new_tensor_view(&cpu_inputs[0..bsize * seq]).unwrap());
        } else {
            self.inputs.as_mut().unwrap().copy_from_host(&cpu_inputs[0..bsize * seq]).unwrap();
        }
        if !cpu_targets.is_empty() {
            if self.targets.is_none() {
                self.targets = Some(ctx.new_tensor_view(&cpu_targets[0..bsize * seq]).unwrap());
            } else {
                self.targets
                    .as_mut()
                    .unwrap()
                    .copy_from_host(&cpu_targets[0..bsize * seq])
                    .unwrap();
            }
        }

        // validate B,T is consistent with how we've allocated the memory before
        assert!(self.batch_size == bsize);
        assert!(self.seq_len == seq);

        // Sync losses from GPU to CPU
        let mut acts = self.acts.as_mut().unwrap().inner();
        let params = self.params.inner();
        let inputs = &mut self.inputs.as_mut().unwrap();
        //acts.encoded, model->inputs, params.wte, params.wpe
        encoder_forward(
            ctx,
            m,
            &mut acts.encoded,
            inputs,
            &params.wte,
            &params.wpe,
            bsize,
            seq,
            ch,
        );

        /*
        let out_len = acts.output.len();
        let mut d_output = vec![0.0f32; out_len];
        */
        let mut ln1w = params.ln1w;
        let mut ln1b = params.ln1b;
        let mut qkvw = params.qkvw;
        let mut qkvb = params.qkvb;
        let mut attprojw = params.attprojw;
        let mut attprojb = params.attprojb;
        let mut ln2w = params.ln2w;
        let mut ln2b = params.ln2b;
        let mut fcw = params.fcw;
        let mut fcb = params.fcb;
        let mut fcprojw = params.fcprojw;
        let mut fcprojb = params.fcprojb;

        let mut ln1 = acts.ln1;
        let mut ln1_mean = acts.ln1_mean;
        let mut ln1_rstd = acts.ln1_rstd;
        let mut qkvr = acts.qkvr;
        let mut atty = acts.atty;
        let mut att = acts.att;
        let mut attproj = acts.attproj;
        let mut residual2 = acts.residual2;
        let mut ln2 = acts.ln2;
        let mut ln2_mean = acts.ln2_mean;
        let mut ln2_rstd = acts.ln2_rstd;
        let mut fch = acts.fch;
        let mut fch_gelu = acts.fch_gelu;
        let mut fcproj = acts.fcproj;
        let num_layers = self.config.num_layers;

        let start = std::time::Instant::now();
        for l in 0..num_layers {
            let mut l_residual = acts.residual3.index_mut(
                if l == 0 { 0 } else { (l - 1) * bsize * seq * ch }..(l + 1) * bsize * seq * ch,
            );
            let (mut res, mut l_residual3) = if l == 0 {
                l_residual.split_at_mut(0)
            } else {
                l_residual.split_at_mut(bsize * seq * ch)
            };
            /*
            float* l_ln1w = params.ln1w + l * C;
            float* l_ln1b = params.ln1b + l * C;
            float* l_qkvw = params.qkvw + l * 3*C * C;
            float* l_qkvb = params.qkvb + l * 3*C;
            float* l_attprojw = params.attprojw + l * C * C;
            float* l_attprojb = params.attprojb + l * C;
            float* l_ln2w = params.ln2w + l * C;
            float* l_ln2b = params.ln2b + l * C;
            float* l_fcw = params.fcw + l * 4*C * C;
            float* l_fcb = params.fcb + l * 4*C;
            float* l_fcprojw = params.fcprojw + l * C * 4*C;
            float* l_fcprojb = params.fcprojb + l * C;

            // get the pointers of the activations for this layer
            float* l_ln1 = acts.ln1 + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            float* l_qkvr = acts.qkvr + l * B * T * 3*C;
            float* l_atty = acts.atty + l * B * T * C;
            float* l_att = acts.att + l * B * NH * T * T;
            float* l_attproj = acts.attproj + l * B * T * C;
            float* l_res2 = acts.residual2 + l * B * T * C;
            float* l_ln2 = acts.ln2 + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch = acts.fch + l * B * T * 4*C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
            float* l_fcproj = acts.fcproj + l * B * T * C;
            float* l_residual3 = acts.residual3 + l * B * T * C;
            // these are only needed as scratchpads for the forward pass, but
            // need not be stored for backward
            float* scratch = acts.output;
            */
            let l_ln1w = ln1w.index_mut(l * ch..(l + 1) * ch);
            let l_ln1b = ln1b.index_mut(l * ch..(l + 1) * ch);
            let l_qkvw = qkvw.index_mut(l * 3 * ch * ch..(l + 1) * 3 * ch * ch);
            let l_qkvb = qkvb.index_mut(l * 3 * ch..(l + 1) * 3 * ch);
            let l_attprojw = attprojw.index_mut(l * ch * ch..(l + 1) * ch * ch);
            let l_attprojb = attprojb.index_mut(l * ch..(l + 1) * ch);
            let l_ln2w = ln2w.index_mut(l * ch..(l + 1) * ch);
            let l_ln2b = ln2b.index_mut(l * ch..(l + 1) * ch);
            let l_fcw = fcw.index_mut(l * 4 * ch * ch..(l + 1) * 4 * ch * ch);
            let l_fcb = fcb.index_mut(l * 4 * ch..(l + 1) * 4 * ch);
            let l_fcprojw = fcprojw.index_mut(l * ch * 4 * ch..(l + 1) * ch * 4 * ch);
            let l_fcprojb = fcprojb.index_mut(l * ch..(l + 1) * ch);

            let mut l_ln1 = ln1.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let mut l_ln1_mean = ln1_mean.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let mut l_ln1_rstd = ln1_rstd.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let mut l_qkvr =
                qkvr.index_mut(l * bsize * seq * 3 * ch..(l + 1) * bsize * seq * 3 * ch);
            let mut l_atty = atty.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let mut l_att =
                att.index_mut(l * bsize * nh * seq * seq..(l + 1) * bsize * nh * seq * seq);
            let mut l_attproj = attproj.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let mut l_res2 = residual2.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let mut l_ln2 = ln2.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let mut l_ln2_mean = ln2_mean.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let mut l_ln2_rstd = ln2_rstd.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let mut l_fch = fch.index_mut(l * bsize * seq * 4 * ch..(l + 1) * bsize * seq * 4 * ch);
            let mut l_fch_gelu =
                fch_gelu.index_mut(l * bsize * seq * 4 * ch..(l + 1) * bsize * seq * 4 * ch);
            let mut l_fcproj = fcproj.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let scratch = &mut acts.output;
            /*
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
            attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_res2, residual, l_attproj, B*T*C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_res2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
            residual_forward(l_residual3, l_res2, l_fcproj, B*T*C);
            */
            layernorm_forward(
                ctx,
                m,
                &mut l_ln1,
                &mut l_ln1_mean,
                &mut l_ln1_rstd,
                if l == 0 { &acts.encoded } else { &mut res },
                &l_ln1w,
                &l_ln1b,
                bsize,
                seq,
                ch,
            );
            matmul_forward(ctx, m, scratch, &l_ln1, &l_qkvw, &l_qkvb, bsize, seq, ch, 3 * ch);
            attention_forward(
                ctx,
                m,
                cublas_handle,
                &mut l_atty,
                &mut l_qkvr,
                &mut l_att,
                scratch,
                bsize,
                seq,
                ch,
                nh,
            );

            matmul_forward(
                ctx,
                m,
                &mut l_attproj,
                &l_atty,
                &l_attprojw,
                &l_attprojb,
                bsize,
                seq,
                ch,
                ch,
            );
            residual_forward(
                ctx,
                m,
                &mut l_res2,
                if l == 0 { &acts.encoded } else { &mut res },
                &l_attproj,
                bsize * seq * ch,
            );
            layernorm_forward(
                ctx,
                m,
                &mut l_ln2,
                &mut l_ln2_mean,
                &mut l_ln2_rstd,
                &l_res2,
                &l_ln2w,
                &l_ln2b,
                bsize,
                seq,
                ch,
            );
            matmul_forward(ctx, m, &mut l_fch, &l_ln2, &l_fcw, &l_fcb, bsize, seq, ch, 4 * ch);
            gelu_forward(ctx, m, &mut l_fch_gelu, &l_fch, bsize * seq * 4 * ch);
            matmul_forward(
                ctx,
                m,
                &mut l_fcproj,
                &l_fch_gelu,
                &l_fcprojw,
                &l_fcprojb,
                bsize,
                seq,
                4 * ch,
                ch,
            );
            residual_forward(ctx, m, &mut l_residual3, &l_res2, &l_fcproj, bsize * seq * ch);
        }
        let residual = &mut acts
            .residual3
            .index_mut((num_layers - 1) * bsize * seq * ch..num_layers * bsize * seq * ch);
        let (lnf, lnf_mean, lnf_rstd) = (&mut acts.lnf, &mut acts.lnf_mean, &mut acts.lnf_rstd);
        let (lnfw, lnfb) = (&params.lnfw, &params.lnfb);
        layernorm_forward(ctx, m, lnf, lnf_mean, lnf_rstd, residual, lnfw, lnfb, bsize, seq, ch);
        let empty_tensor = &ctx.new_tensor_view([].as_slice()).unwrap();
        let mut acts = self.acts.as_mut().unwrap().inner();
        matmul_forward(
            ctx,
            m,
            &mut acts.output,
            &acts.lnf,
            &params.wte,
            empty_tensor,
            bsize,
            seq,
            ch,
            pad_vocab,
        );

        if cpu_targets.is_empty() {
            self.mean_loss = -1.0;
            return;
        }
        let targets = &self.targets.as_mut().unwrap();
        fused_classifier3(
            ctx,
            m,
            &mut acts.output,
            &mut acts.losses,
            empty_tensor,
            targets,
            bsize,
            seq,
            self.config.vocab_size,
            pad_vocab,
        );
        if self.cpu_losses.is_none() {
            self.cpu_losses = Some(PinnedHostBox::new_from_tensor(ctx, &acts.losses).unwrap());
        } else {
            acts.losses.copy_to_host(self.cpu_losses.as_mut().unwrap()).unwrap();
        }
        let mean_loss = self.cpu_losses.as_ref().unwrap()[0..(bsize * seq)].iter().sum::<f32>()
            / (bsize * seq) as f32;
        self.mean_loss = mean_loss;
        info!("time for final layer and loss: {:?}", start.elapsed());
        println!("mean loss: {}", mean_loss);
    }
    pub fn zero_grad(&mut self) {
        if let Some(grads) = &mut self.grads {
            grads.tensor.memset(0).expect("failed to zero grads");
        }
        if let Some(grads_acts) = &mut self.grads_acts {
            grads_acts.tensor.memset(0).expect("failed to zero grads_acts");
        }
    }

    pub fn backward(&mut self, cublas_handle: cublas_sys::cublasHandle_t) {
        let ctx = self.ctx;
        if self.mean_loss == -1.0 {
            panic!("No loss computed, cannot run backward");
        }

        if self.grads.is_none() {
            let grads_params = self.config.get_params_sizes();
            let num_parameters: usize = grads_params.iter().sum();
            // TODO: optimize: could just zero the grads instead of copy
            self.grads = Some(ParameterTensors::new(ctx, grads_params, &vec![0.0; num_parameters]));
            println!(
                "allocated {} MiB for parameter gradients\n",
                (num_parameters * std::mem::size_of::<f32>()) >> 20,
            );
        }

        if self.grads_acts.is_none() {
            let grads_acts_sizes = self.config.get_grad_act_sizes(self.batch_size, self.seq_len);
            let num_grad_acts: usize = grads_acts_sizes.iter().sum();
            // TODO: optimize: could just zero the grads instead of copy
            self.grads_acts =
                Some(GradActTensors::new(ctx, grads_acts_sizes, &vec![0.0; num_grad_acts]));
            println!(
                "allocated {} MiB for activation gradients\n",
                (num_grad_acts * std::mem::size_of::<f32>()) >> 20,
            );
        }

        /*
        int B = model->batch_size;
        int T = model->seq_len;
        int Vp = model->config.padded_vocab_size;
        int L = model->config.num_layers;
        int NH = model->config.num_heads;
        int C = self->config.channels;
        */
        let bsize = self.batch_size;
        let seq = self.seq_len;
        let pad_vocab = self.config.padded_vocab_size;
        let num_layers = self.config.num_layers;
        let nh = self.config.num_heads;
        let ch = self.config.channels;
        let m = self.module;

        let mut grads_acts = self.grads_acts.as_mut().unwrap().inner();
        let mut grads = self.grads.as_mut().unwrap().inner();
        let mut acts = self.acts.as_mut().unwrap().inner();
        let mut params = self.params.inner();
        // matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
        matmul_backward(
            ctx,
            m,
            cublas_handle,
            &mut grads_acts.bt4c,
            &mut grads.wte,
            None,
            &acts.output,
            &acts.lnf,
            &params.wte,
            bsize,
            seq,
            ch,
            pad_vocab,
        );
        /*
            float* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
        float* dresidual = grads_acts.residual3; // the main buffer holding the gradient in the backward pass
        layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);
             */

        let residual = acts
            .residual3
            .index_mut((num_layers - 1) * bsize * seq * ch..(num_layers * bsize * seq * ch));
        layernorm_backward(
            ctx,
            m,
            &mut grads_acts.residual3,
            &mut grads.lnfw,
            &mut grads.lnfb,
            &grads_acts.bt4c,
            &residual,
            &params.lnfw,
            &acts.lnf_mean,
            &acts.lnf_rstd,
            bsize,
            seq,
            ch,
        );
        let btc_len = bsize * seq * ch;
        let mut dresidual = grads_acts.residual3;
        // ok here.
        for l in (0..num_layers).rev() {
            // residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
            let residual = if l == 0 {
                &acts.encoded
            } else {
                &acts.residual3.index((l - 1) * bsize * seq * ch..l * bsize * seq * ch)
            };
            let ln1w = params.ln1w.index_mut(l * ch..(l + 1) * ch);
            let qkvw = params.qkvw.index_mut(l * 3 * ch * ch..(l + 1) * 3 * ch * ch);
            let attprojw = params.attprojw.index_mut(l * ch * ch..(l + 1) * ch * ch);
            let ln2w = params.ln2w.index_mut(l * ch..(l + 1) * ch);
            let fcw = params.fcw.index_mut(l * 4 * ch * ch..(l + 1) * 4 * ch * ch);
            let fcprojw = params.fcprojw.index_mut(l * ch * 4 * ch..(l + 1) * ch * 4 * ch);

            // get the pointers of the gradients of the weights for this layer
            // float* dl_ln1w = grads.ln1w + l * C;
            // float* dl_ln1b = grads.ln1b + l * C;
            // float* dl_qkvw = grads.qkvw + l * 3*C * C;
            // float* dl_qkvb = grads.qkvb + l * 3*C;
            // float* dl_attprojw = grads.attprojw + l * C * C;
            // float* dl_attprojb = grads.attprojb + l * C;
            // float* dl_ln2w = grads.ln2w + l * C;
            // float* dl_ln2b = grads.ln2b + l * C;
            // float* dl_fcw = grads.fcw + l * 4*C * C;
            // float* dl_fcb = grads.fcb + l * 4*C;
            // float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
            // float* dl_fcprojb = grads.fcprojb + l * C;
            let mut dl_ln1w = grads.ln1w.index_mut(l * ch..(l + 1) * ch);
            let mut dl_ln1b = grads.ln1b.index_mut(l * ch..(l + 1) * ch);
            let mut dl_qkvw = grads.qkvw.index_mut(l * 3 * ch * ch..(l + 1) * 3 * ch * ch);
            let mut dl_qkvb = grads.qkvb.index_mut(l * 3 * ch..(l + 1) * 3 * ch);
            let mut dl_attprojw = grads.attprojw.index_mut(l * ch * ch..(l + 1) * ch * ch);
            let mut dl_attprojb = grads.attprojb.index_mut(l * ch..(l + 1) * ch);
            let mut dl_ln2w = grads.ln2w.index_mut(l * ch..(l + 1) * ch);
            let mut dl_ln2b = grads.ln2b.index_mut(l * ch..(l + 1) * ch);
            let mut dl_fcw = grads.fcw.index_mut(l * 4 * ch * ch..(l + 1) * 4 * ch * ch);
            let mut dl_fcb = grads.fcb.index_mut(l * 4 * ch..(l + 1) * 4 * ch);
            let mut dl_fcprojw = grads.fcprojw.index_mut(l * ch * 4 * ch..(l + 1) * ch * 4 * ch);
            let mut dl_fcprojb = grads.fcprojb.index_mut(l * ch..(l + 1) * ch);

            // get the pointers of the activations for this layer
            // float* l_ln1 = acts.ln1 + l * B * T * C;
            // float* l_ln1_mean = acts.ln1_mean + l * B * T;
            // float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            // float* l_qkvr = acts.qkvr + l * B * T * 3*C;
            // float* l_atty = acts.atty + l * B * T * C;
            // float* l_att = acts.att + l * B * NH * T * T;
            // float* l_residual2 = acts.residual2 + l * B * T * C;
            // float* l_ln2 = acts.ln2 + l * B * T * C;
            // float* l_ln2_mean = acts.ln2_mean + l * B * T;
            // float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            // float* l_fch = acts.fch + l * B * T * 4*C;
            // float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
            let ln1 = acts.ln1.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let ln1_mean = acts.ln1_mean.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let ln1_rstd = acts.ln1_rstd.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let qkvr =
                acts.qkvr.index_mut(l * bsize * seq * 3 * ch..(l + 1) * bsize * seq * 3 * ch);
            let mut atty = acts.atty.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let att =
                acts.att.index_mut(l * bsize * nh * seq * seq..(l + 1) * bsize * nh * seq * seq);
            let residual2 =
                acts.residual2.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let ln2 = acts.ln2.index_mut(l * bsize * seq * ch..(l + 1) * bsize * seq * ch);
            let ln2_mean = acts.ln2_mean.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let ln2_rstd = acts.ln2_rstd.index_mut(l * bsize * seq..(l + 1) * bsize * seq);
            let mut fch =
                acts.fch.index_mut(l * bsize * seq * 4 * ch..(l + 1) * bsize * seq * 4 * ch);
            let fch_gelu =
                acts.fch_gelu.index_mut(l * bsize * seq * 4 * ch..(l + 1) * bsize * seq * 4 * ch);

            // get the pointers of the gradients of the activations for this layer
            // notice that there is no l *, because we just have a single copy, and keep
            // re-using this memory in every Transformer block as we calculate backward pass

            // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
            // so we can co-opt it here.
            // float* dl_btc = acts.lnf;
            // float* dl_bt4c = grads_acts.bt4c;
            // float* dl_preatt = grads_acts.preatt;
            let dl_btc = &mut acts.lnf;
            let dl_bt4c = &mut grads_acts.bt4c;
            let dl_preatt = &mut grads_acts.preatt;

            // re-use scratch buffer of the forward pass
            // float* scratch = acts.output;
            let scratch = &mut acts.output;
            // backprop this layer
            // matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
            // gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
            // matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
            // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
            // layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            // matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
            // we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
            // wrong dl_fcprojw.
            matmul_backward(
                ctx,
                m,
                cublas_handle,
                dl_bt4c,
                &mut dl_fcprojw,
                Some(&mut dl_fcprojb),
                &mut dresidual,
                &fch_gelu,
                &fcprojw,
                bsize,
                seq,
                4 * ch,
                ch,
            );
            // correct until here.
            /*
            println!("bt_bt4c\n{}", dl_bt4c);
            println!("dl_fcprojw\n{}", dl_fcprojw);
            println!("dl_fcprojb\n{}", dl_fcprojb);
            println!("dresidual\n{}", dresidual);
            println!("fch_gelu\n{}", fch_gelu);
            println!("fcprojw\n{}", fcprojw);
            panic!();
            */
            // gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
            //println!("dl_bt4c\n{}", dl_bt4c);
            gelu_backward(ctx, m, dl_bt4c, &fch, bsize * seq * 4 * ch);
            // correct until here.
            /*println!("dl_bt4c\n{}", dl_bt4c);
            println!("fch\n{}", fch);
            panic!();*/
            // matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
            matmul_backward(
                ctx,
                m,
                cublas_handle,
                dl_btc,
                &mut dl_fcw,
                Some(&mut dl_fcb),
                dl_bt4c,
                &ln2,
                &fcw,
                bsize,
                seq,
                ch,
                4 * ch,
            );
            //layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            layernorm_backward(
                ctx,
                m,
                &mut dresidual,
                &mut dl_ln2w,
                &mut dl_ln2b,
                dl_btc,
                &residual2,
                &ln2w,
                &ln2_mean,
                &ln2_rstd,
                bsize,
                seq,
                ch,
            );
            // matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
            matmul_backward(
                ctx,
                m,
                cublas_handle,
                dl_btc,
                &mut dl_attprojw,
                Some(&mut dl_attprojb),
                &dresidual,
                &atty,
                &attprojw,
                bsize,
                seq,
                ch,
                ch,
            );
            // float* buffer_a = l_atty;
            // float* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need
            // attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
            // matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
            // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
            // layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
            /*println!("backward updated: dl_fcprojw\n{}", dl_fcprojw);
            println!("backward updated: dl_fcw\n{}", dl_fcw);
            println!("backward updated: dl_attprojw\n{}", dl_attprojw);
            println!("backward updated: dl_qkvw\n{}", dl_qkvw);
            println!("backward updated: dl_ln2w\n{}", dl_ln2w);
            println!("backward updated: dl_ln1w\n{}", dl_ln1w);
            println!("backward updated: dresidual\n{}", dresidual);
            panic!();*/
            attention_backward(
                ctx,
                m,
                cublas_handle,
                dl_bt4c,
                &mut fch,
                dl_preatt,
                scratch,
                &mut atty,
                dl_btc,
                &qkvr,
                &att,
                bsize,
                seq,
                ch,
                nh,
            );
            /*println!("dl_bt4c\n{}", dl_bt4c);
            println!("fch\n{}", fch);
            println!("dl_preatt\n{}", dl_preatt);
            println!("scratch\n{}", scratch);
            println!("atty\n{}", atty);
            println!("dl_btc\n{}", dl_btc);
            println!("qkvr\n{}", qkvr);
            println!("att\n{}", att);
            panic!();*/
            // matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
            matmul_backward(
                ctx,
                m,
                cublas_handle,
                dl_btc,
                &mut dl_qkvw,
                Some(&mut dl_qkvb),
                dl_bt4c,
                &ln1,
                &qkvw,
                bsize,
                seq,
                ch,
                3 * ch,
            );
            layernorm_backward(
                ctx,
                m,
                &mut dresidual,
                &mut dl_ln1w,
                &mut dl_ln1b,
                dl_btc,
                residual,
                &ln1w,
                &ln1_mean,
                &ln1_rstd,
                bsize,
                seq,
                ch,
            );
        }
        // encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
        let inputs = &mut self.inputs.as_mut().unwrap();
        encoder_backward(
            ctx,
            m,
            &mut grads.wte,
            &mut grads.wpe,
            &dresidual,
            inputs,
            bsize,
            seq,
            ch,
        );
        /*println!("wte\n{}", grads.wte);
        println!("wpe\n{}", grads.wpe);
        println!("dresidual\n{}", dresidual);
        println!("inputs\n{}", inputs);
        panic!();*/
    }

    pub fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
    ) {
        let ctx = self.ctx;
        let m = self.module;
        if self.grads.is_none() {
            panic!("No gradients allocated, cannot run update");
        }
        let num_parameters = self.params.tensor.len();
        if self.m_memory.is_none() {
            self.m_memory =
                Some(ctx.new_tensor_view(vec![0.0; num_parameters].as_slice()).unwrap());
            self.v_memory =
                Some(ctx.new_tensor_view(vec![0.0; num_parameters].as_slice()).unwrap());
            println!(
                "allocated {} MiB for AdamW optimizer state m\n",
                (num_parameters * std::mem::size_of::<f32>()) >> 20
            );
            println!(
                "allocated {} MiB for AdamW optimizer state v\n",
                (num_parameters * std::mem::size_of::<f32>()) >> 20
            );
        }
        let m_memory = &mut self.m_memory.as_mut().unwrap();
        let v_memory = &mut self.v_memory.as_mut().unwrap();
        let param_memory = &mut self.params.tensor;
        let grads_memory = &self.grads.as_ref().unwrap().tensor;
        const BSIZE: usize = 512;

        let num_blocks = num_parameters.div_ceil(BSIZE) as u32;
        let beta1_correction = 1.0f32 - beta1.powi(step);
        let beta2_correction = 1.0f32 - beta2.powi(step);
        let config = gpu_host::gpu_config!(num_blocks, 1, 1, @const BSIZE as u32, 1, 1, 0);
        llm_rs_gpu::adamw_kernel2::launch(
            config,
            ctx,
            m,
            param_memory,
            grads_memory,
            m_memory,
            v_memory,
            num_parameters as i32,
            learning_rate,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            weight_decay,
        )
        .expect("Failed to launch adamw kernel");
        /*println!("param_memory\n{}", param_memory);
        println!("grads_memory\n{}", grads_memory);
        println!("m_memory\n{}", m_memory);
        println!("v_memory\n{}", v_memory);
        panic!();*/
    }
}
