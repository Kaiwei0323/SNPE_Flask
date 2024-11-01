// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include "SNPERuntime.h"

namespace snperuntime {

    /** @brief SNPE constructor
    */
    SNPERuntime::SNPERuntime()
    {
        static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
        LOG_INFO("Using SNPE: '%s' \n", version.asString().c_str());
        m_isInit = false;
    }

    SNPERuntime::~SNPERuntime()
    {

    }

    bool SNPERuntime::IsInit(void) 
    {
        return m_isInit;
    }

    /** @brief To calculate buffer size for memory allocation
     * @return buffer size
    */
    static size_t calcSizeFromDims(const zdl::DlSystem::Dimension* dims, size_t rank, size_t elementSize)
    {
        if (rank == 0) return 0;
        size_t size = elementSize;
        while (rank--) {
            size *= *dims;
            dims++;
        }
        return size;
    }

    /** @brief To create userbuffer
    */
    void CreateUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
        std::unordered_map<std::string, std::vector<float>>& applicationBuffers,
        std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
        const zdl::DlSystem::TensorShape& bufferShape,
        const char* name)
    {
        size_t bufferElementSize = sizeof(float);

        /**
        * To calculate stride based on buffer strides
        * Note: Strides = Number of bytes to advance to the next element in each dimension.
        * For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
        */
        std::vector<size_t> strides(bufferShape.rank());
        strides[strides.size() - 1] = bufferElementSize;
        size_t stride = strides[strides.size() - 1];
        for (size_t i = bufferShape.rank() - 1; i > 0; i--)
        {
            stride *= bufferShape[i];
            strides[i - 1] = stride;
        }
        size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);
        /**
         * To set the buffer encoding type
        */
        zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
        /**
         * To create user-backed storage to load input data onto it
        */
        applicationBuffers.emplace(name, std::vector<float>(bufSize / bufferElementSize));
        /**
         * To create SNPE user buffer from the user-backed buffer
        */
        zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
        snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer((void*)applicationBuffers.at(name).data(),
            bufSize,
            strides,
            &userBufferEncodingFloat));
        /**
         * To add the user-backed buffer to the inputMap, which is later on fed to the network for execution
        */
        
        if (snpeUserBackedBuffers.back() == nullptr)
        {
            std::cerr << "Error while creating user buffer." << std::endl;
        }
        userBufferMap.add(name, snpeUserBackedBuffers.back().get());
    }

    /** @brief To set SNPERuntime
     * @param runtime contains SNPERuntime value
    */
    void SNPERuntime::setTargetRuntime(runtime_t& runtime)
    {
        switch (runtime) {
        case DSP:
            m_runtime = zdl::DlSystem::Runtime_t::DSP;
            break;
        case GPU:
            m_runtime = zdl::DlSystem::Runtime_t::GPU;
            break;
        case GPU_FLOAT16:
            m_runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
            break;
        case AIP_FIXED_TF:
            m_runtime = zdl::DlSystem::Runtime_t::AIP_FIXED_TF;
            break;
        case CPU:
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            break;
        default:
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            break;
        }

        if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(m_runtime)) {
            LOG_ERROR("Selected runtime not present. Falling back to CPU.\n");
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            runtime = CPU;
        }
    }

    /** @brief To set performance profile
     *  @param perfprofile contains performance value
    */
    void SNPERuntime::setPerformanceProfile(const performance_t perfprofile)
    {
        switch (perfprofile) {
        case BALANCED:
            m_profile = zdl::DlSystem::PerformanceProfile_t::BALANCED;
            break;
        case HIGH_PERFORMANCE:
            m_profile = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;
            break;
        case POWER_SAVER:
            m_profile = zdl::DlSystem::PerformanceProfile_t::POWER_SAVER;
            break;
        case SUSTAINED_HIGH_PERFORMANCE:
            m_profile = zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE;
            break;
        case BURST:
            m_profile = zdl::DlSystem::PerformanceProfile_t::BURST;
            break;
        case LOW_POWER_SAVER:
            m_profile = zdl::DlSystem::PerformanceProfile_t::LOW_POWER_SAVER;
            break;
        case HIGH_POWER_SAVER:
            m_profile = zdl::DlSystem::PerformanceProfile_t::HIGH_POWER_SAVER;
            break;
        case LOW_BALANCED:
            m_profile = zdl::DlSystem::PerformanceProfile_t::LOW_BALANCED;
            break;
        case SYSTEM_SETTINGS:
            m_profile = zdl::DlSystem::PerformanceProfile_t::SYSTEM_SETTINGS;
            break;
        default:
            m_profile = zdl::DlSystem::PerformanceProfile_t::BALANCED;
            break;
        }
#ifdef DEBUG
        LOG_DEBUG("Chosen performance: %d,  Set performance: %d \n", perfprofile, (int)m_profile);
#endif
    }

    /** @brief To initialize SNPERuntime
     * @param dlc_path contains dlc path from the config file
     * @param runtime SNPERuntime value
     * @return true if success; false otherwise
    */
    bool SNPERuntime::Initialize(const std::string& dlc_path, runtime_t runtime, performance_t profiling_level, bool enable_init_cache)
    {
        setTargetRuntime(runtime);
        setPerformanceProfile(profiling_level);
        /**
         * To read dlc from dlc_path
        */
        m_container = zdl::DlContainer::IDlContainer::open(dlc_path);
        if (m_container == NULL)
        {
            LOG_ERROR("Error: Failed to open %s\n", dlc_path.c_str());
        }
        /**
         * To create snpeBuilder from m_container based on runtime,performance profile
        */
        std::vector<std::string> runtimeStrVector;
        switch (runtime)
        {
            case CPU:
                runtimeStrVector.push_back("cpu_float32");
                LOG_INFO("*** CPU runtime***\n");
                break;
                
            case DSP:
                runtimeStrVector.push_back("dsp_fixed8_tf");
                LOG_INFO("*** DSP runtime***\n");
                break;

            case GPU:
                runtimeStrVector.push_back("gpu_float32_16_hybrid");
                LOG_INFO("*** GPU runtime***\n");
                break;

            case GPU_FLOAT16:
                runtimeStrVector.push_back("gpu_float32_16_hybrid");
                LOG_INFO("*** GPU FLOAT16 runtime***\n");
                break;

            case AIP_FIXED_TF:
                runtimeStrVector.push_back("aip_fixed8_tf");
                LOG_INFO("*** AIP FIXED TF runtime***\n");
                break;

            default:
                LOG_ERROR("Unknown runtime type, defaulting to CPU.\n");
                runtimeStrVector.push_back("cpu_float32");
                break;
        }

        zdl::DlSystem::RuntimeList runtimeList;

        runtimeList.clear();
        for (auto& runtimeStr : runtimeStrVector)
        {
            zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::RuntimeList::stringToRuntime(runtimeStr.c_str());
            if (runtime != zdl::DlSystem::Runtime_t::UNSET)
            {
                bool ret = runtimeList.add(runtime);
                if (ret == false)
                {
                    LOG_ERROR("Error: runtime order of precedence\n");
                    return false;
                }
            }
            else
            {
                LOG_ERROR("Error: Invalid values passed to the runtime order of precedence\n");
                return false;
            }
        }
        zdl::SNPE::SNPEBuilder snpeBuilder(m_container.get());
        m_snpe = snpeBuilder.setOutputLayers(m_outputLayers)
            .setPerformanceProfile(m_profile)
            .setUseUserSuppliedBuffers(true)
            .setRuntimeProcessorOrder(runtimeList)
            .setInitCacheMode(enable_init_cache)
            .build();

        if (nullptr == m_snpe.get()) {
            const char* errStr = zdl::DlSystem::getLastErrorString();
            LOG_ERROR("SNPE build failed: {%s}\n", errStr);
            return false;
        }

        /**
         *  To get input tensor names of the network that needs to be populated
        */
        const auto& inputNamesOpt = m_snpe->getInputTensorNames();
        if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names\n");
        const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
        /**
         *  To create SNPE user buffers for each application storage buffer
        */
        for (const char* name : inputNames) {
            /**
             * To get attributes of buffer by name
            */
            auto bufferAttributesOpt = m_snpe->getInputOutputBufferAttributes(name);
            if (!bufferAttributesOpt) {
                LOG_ERROR("Error obtaining attributes for input tensor: %s\n", name);
                return false;
            }

            const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
            std::vector<size_t> tensorShape;
            for (size_t j = 0; j < bufferShape.rank(); j++) {
                tensorShape.push_back(bufferShape[j]);
            }
            m_inputShapes.emplace(name, tensorShape);
            CreateUserBuffer(m_inputUserBufferMap, m_applicationInputBuffers, m_inputUserBuffers, bufferShape, name);
        }
        /**
         * To get output tensor names of the network that need to be populated
        */
        const auto& outputNamesOpt = m_snpe->getOutputTensorNames();
        if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names\n");
        const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

        /**
         *  To create SNPE user buffers for each application storage buffer
        */
        for (const char* name : outputNames) {
            // get attributes of buffer by name
            auto bufferAttributesOpt = m_snpe->getInputOutputBufferAttributes(name);
            if (!bufferAttributesOpt) {
                LOG_ERROR("Error obtaining attributes for input tensor: %s\n", name);
                return false;
            }

            const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
            std::vector<size_t> tensorShape;
            for (size_t j = 0; j < bufferShape.rank(); j++) {
                tensorShape.push_back(bufferShape[j]);
            }
            m_outputShapes.emplace(name, tensorShape);

            CreateUserBuffer(m_outputUserBufferMap, m_applicationOutputBuffers, m_outputUserBuffers, bufferShape, name);
        }

        m_isInit = true;

        return true;
    }

    /** @brief To deinitialize SNPERuntime
    */
    bool SNPERuntime::Deinitialize(void)
    {
        if (nullptr != m_snpe) {
            m_snpe.reset(nullptr);
        }

        for (auto [k, v] : m_applicationInputBuffers) ClearVector(v);
        for (auto [k, v] : m_applicationOutputBuffers)  ClearVector(v);
        return true;
    }

    /** @brief To store output layers for each model
     * @param outputlayers contains output layers defined in the config file
    */
    bool SNPERuntime::SetOutputLayers(std::vector<std::string>& outputLayers)
    {
        for (size_t i = 0; i < outputLayers.size(); i++) 
        {
            m_outputLayers.append(outputLayers[i].c_str());
        }

        return true;
    }

    /** @brief To get input shape for each model
     * @param name contains name of input layer
     * @return shape of input layer if success; empty otherwise
    */
    std::vector<size_t> SNPERuntime::GetInputShape(const std::string& name)
    {
        /**
         * To check if runtime is initialized and layer name is a part of input
        */
        if (IsInit()) {
            if (m_inputShapes.find(name) != m_inputShapes.end()) 
            {
                return m_inputShapes.at(name);
            }
            LOG_ERROR("Can't find any input layer named %s\n", name.c_str());
            return {};
        }
        else {
            LOG_ERROR("GetInputShape Failed: SNPE Init Failed !!!\n");
            return {};
        }
    }

    /** @brief To get output shape for each model
     * @param name contains name of output layers
     * @return shape of output layer if success; empty otherwise
    */
    std::vector<size_t> SNPERuntime::GetOutputShape(const std::string& name)
    {
        /**
        * To check if runtime is initialized and layer name is a part of output
       */
        if (IsInit()) {
            if (m_outputShapes.find(name) != m_outputShapes.end()) 
            {
                return m_outputShapes.at(name);
            }
            LOG_ERROR("Can't find any ouput layer named %s\n", name.c_str());
            return {};
        }
        else 
        {
            LOG_ERROR("GetOutputShape Failed: SNPE Init Failed !!!\n");
            return {};
        }
    }


    /** @brief To get input tensor for each model
     * @param name contains name of input layer
     * @return shape of input tensor if success; NULL otherwise
    */
    float* SNPERuntime::GetInputTensor(const std::string& name)
    {
        /**
        * To check if runtime is initialized and layer name is a part of input
       */
        if (IsInit()) 
        {
            if (m_applicationInputBuffers.find(name) != m_applicationInputBuffers.end()) 
            {
                return m_applicationInputBuffers.at(name).data();
            }
            LOG_ERROR("Can't find any input tensor named '%s' \n", name.c_str());
            return nullptr;
        }
        else 
        {
            LOG_ERROR("GetInputTensor Failed: SNPE Init Failed !!!\n");
            return nullptr;
        }
    }

    /** @brief To get output tensor for each model
     * @param name contains name of output layer
     * @return shape of output tensor if success; NULL otherwise
    */

    float* SNPERuntime::GetOutputTensor(const std::string& name)
    {
        /**
        * To check if runtime is initialized and layer name is a part of output
       */
        if (IsInit()) {
            if (m_applicationOutputBuffers.find(name) != m_applicationOutputBuffers.end()) 
            {
                return m_applicationOutputBuffers.at(name).data();
            }
            LOG_ERROR("Can't find any output tensor named '%s' \n", name.c_str());
            return nullptr;
        }
        else 
        {
            LOG_ERROR("GetOutputTensor Failed: SNPE Init Failed !!!");
            return nullptr;
        }
    }

    /** @brief To execute inference on target
     * @return QS_SUCCESS if success; QS_FAIL otherwise
    */
    bool SNPERuntime::execute()
    {
        if (!m_snpe->execute(m_inputUserBufferMap, m_outputUserBufferMap))
        {
            LOG_ERROR("SNPE Task execute failed: %s\n", zdl::DlSystem::getLastErrorString());
            return false;
        }

        return true;
    }

}   // namespace snperuntime
