//
//  LLMEvaluator.swift
//  LLMEval
//
//  Created by Avimanyu Roy on 08/04/24.
//

import Observation
import Tokenizers
import LLM
import MLX
import MLXRandom
import SwiftUI


@Observable
class LLMEvaluator {

    @MainActor
    var running = false

    var output = ""
    var modelInfo = ""
    var stat = ""

    /// this controls which model loads -- phi4bit is one of the smaller ones so this will fit on
    /// more devices
    #if os(macOS)
    let modelConfiguration = ModelConfiguration.llama3IT
    #else
    let modelConfiguration = ModelConfiguration.miniCPM
    #endif
    /// parameters controlling the output
    let temperature: Float = 0.5
    var maxTokens = 8000

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 10

    enum LoadState {
        case idle
        case loaded(LLMModel, Tokenizers.Tokenizer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> (LLMModel, Tokenizers.Tokenizer) {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let (model, tokenizer) = try await LLM.load(configuration: modelConfiguration) {
                [modelConfiguration] progress in
                DispatchQueue.main.sync {
                    self.modelInfo =
                        "Downloading \(modelConfiguration.id): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(MLX.GPU.activeMemory / 1024 / 1024)M"
            loadState = .loaded(model, tokenizer)
            return (model, tokenizer)

        case .loaded(let model, let tokenizer):
            return (model, tokenizer)
        }
    }

    func generate(prompt: String) async {
        let startTime = Date()
        do {
            let (model, tokenizer) = try await load()
            
            await MainActor.run {
                self.running = true
                self.output = ""
            }
            
            // augment the prompt as needed
            let prompt = modelConfiguration.prepare(prompt: prompt)
            let promptTokens = MLXArray(tokenizer.encode(text: prompt))
            
            var initTime = Date()
            let initDuration = initTime.timeIntervalSince(startTime)
            await MainActor.run {
                self.stat = "Init: \(String(format: "%.3f", initDuration))s"
            }
            
            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
            
            var outputTokens = [Int]()
            

            for token in TokenIterator(prompt: promptTokens, model: model, temp: temperature) {
                let tokenId = token.item(Int.self)

                // to match the measurement from the command line we reset the start time
                // after the first token is generated (called the prompt time)
                if outputTokens.isEmpty {
                    initTime = Date()
                }

                if tokenId == tokenizer.unknownTokenId || tokenId == tokenizer.eosTokenId {
                    break
                }
                

                outputTokens.append(tokenId)
                let text = tokenizer.decode(tokens: outputTokens)

                // update the output -- this will make the view show the text as it generates
                if outputTokens.count % displayEveryNTokens == 0 {
                    await MainActor.run {
                        self.output = text
                    }
                }

                if outputTokens.count >= maxTokens {
                    break
                }
            }

            let tokenDuration = Date().timeIntervalSince(initTime)
            let tokensPerSecond = Double(outputTokens.count) / tokenDuration

            // update the text if needed, e.g. we haven't displayed because of displayEveryNTokens
            let finalText = tokenizer.decode(tokens: outputTokens)

            await MainActor.run {
                if finalText != self.output {
                    self.output = finalText
                }
                running = false
                self.stat += " Tokens/second: \(String(format: "%.3f", tokensPerSecond))"
            }

        } catch {
            await MainActor.run {
                running = false
                output = "Failed: \(error)"
            }
        }
    }
}
